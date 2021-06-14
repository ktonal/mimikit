import h5py
import pandas as pd
from multiprocessing import cpu_count, Pool
from concurrent.futures import ThreadPoolExecutor
import os
import warnings

from .regions import Regions

__all__ = [
    'make_root_db',
    'write_feature'
]

warnings.filterwarnings("ignore", message="PySoundFile failed.")


def _sizeof_fmt(num, suffix='b'):
    for unit in ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


# Core function

def write_feature(h5_file, feature_name, attrs, data, regions=None, files=None):
    with h5py.File(h5_file, "r+") as f:
        if feature_name in f:
            f.pop(feature_name)
        ds = f.create_dataset(name=feature_name, shape=data.shape, data=data)
        ds.attrs.update(attrs)
        f.attrs["features"] = [*f.attrs["features"], feature_name]
    if regions is not None:
        pd.DataFrame(regions).to_hdf(h5_file, feature_name + "_regions", "r+")
    if files is not None:
        pd.DataFrame(files).to_hdf(h5_file, feature_name + "_files", "r+")
    return None


def file_to_h5(abs_path, extract_func=None, output_path=None, mode="w"):
    """
    apply ``extract_func`` to ``abs_path`` and write the result in a .h5 file.

    Parameters
    ----------
    abs_path : str
        path to the file to be extracted
    extract_func : function
        the function to use for the extraction - should take exactly one argument: the path to the file to be extracted.
    output_path : str or None
        name of the created .h5 file if not None, else the name of the created .h5 file
        will be of the form : "<abs_path_without_extension>.h5"
    mode : str
        the mode to use when opening the .h5 file. default is "w".

    Returns
    -------
    created_file, infos : str, dict
        the name of the created .h5 file and a ``dict`` with the keys ``"dtype"`` and ``"shape"``
    """
    # print("making .h5 for %s" % abs_path)
    if output_path is None:
        output_path = os.path.splitext(abs_path)[0] + ".h5"
    else:
        if output_path[-3:] != ".h5":
            output_path += ".h5"
    # print("!!!", abs_path)
    rv = extract_func(abs_path)
    info = {}
    if os.path.exists(output_path) and mode == 'w':
        os.remove(output_path)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        f = h5py.File(output_path, mode)
    except OSError as e:
        print(output_path)
        raise e
    for name, (attrs, data, regions) in rv.items():
        ds = f.create_dataset(name=name, shape=data.shape, data=data)
        ds.attrs.update(attrs)
        info[name] = {"dtype": ds.dtype, "shape": ds.shape}
        if regions is not None:
            f.close()
            pd.DataFrame(regions).to_hdf(output_path, name + "_regions", "r+")
            f = h5py.File(output_path, "r+")
        del data
    f.attrs["features"] = list(rv.keys())
    f.flush()
    f.close()
    del rv
    return output_path, info


# Multiprocessing routine

def _make_db_for_each_file(file_walker,
                           extract_func=None,
                           destination="",
                           n_cores=cpu_count(),
                           method='mp'):
    """
    apply ``extract_func`` to the files found by ``file_walker``

    If file_walker found more than ``n_cores`` files, a multiprocessing ``Pool`` is used to speed up the process.

    Parameters
    ----------
    file_walker : iterable of str
        collection of files to be processed
    extract_func : function
        the function to apply to each file. Must take only one argument (the path to the file)
    n_cores : int, optional
        the number of cores used in the ``Pool``, default is the number of available cores on the system.

    Returns
    -------
    temp_dbs : list of tuples
        each tuple in the list is of the form ``("<created_file>.h5", dict(feature_name=dict(dtype=..., shape=...), ...))``
    """
    # add ".tmp_" prefix to the output_paths
    # args = [(file, extract_func, )
    args = [(file, extract_func, os.path.join(destination, file.strip('.').strip('/')))
            for n, file in enumerate(file_walker)]
    if len(args) > 1:
        if method != 'future':
            with Pool(min(n_cores, len(args))) as p:
                tmp_dbs_infos = p.starmap(file_to_h5, args)
        else:
            with ThreadPoolExecutor(max_workers=len(args)) as executor:
                tmp_dbs_infos = [info for info in executor.map(file_to_h5, *zip(*args))]
    else:
        tmp_dbs_infos = [file_to_h5(*arg) for arg in args]
    return tmp_dbs_infos


def _aggregate_db_infos(infos):
    """
    helper to reduce shapes and dtypes of several files and to concatenate their regions

    Parameters
    ----------
    infos : list of tuples
        see the returned value of ``_make_db_for_each_file`` for the expected type

    Returns
    -------
    ds_definitions : dict
        the keys are the name of the H5Datasets (features)
        the values are dictionaries with keys ("shape", "dtype", "regions") and corresponding values
    """
    paths = [x[0] for x in infos]
    features = set([feature for db in infos for feature in db[1].keys()])
    ds_definitions = {}
    for f in features:
        dtype = set([db[1][f]["dtype"] for db in infos])
        assert len(dtype) == 1, "aggregated features must be of a unique dtype"
        dtype = dtype.pop()
        shapes = [db[1][f]["shape"] for db in infos]
        dims = shapes[0][1:]
        if not all(shp[1:] == dims for shp in shapes[1:]):
            raise ValueError("all features should have the same dimensions but for the first axis." + \
                             f" feature {str(f)} returned different shapes")
        # collect the regions for the files (this is different from the possible segmentation regions!)
        regions = Regions.from_duration([s[0] for s in shapes])
        ds_shape = (regions.stop.values[-1], *dims)
        regions.index = paths
        ds_definitions[f] = {"shape": ds_shape, "dtype": dtype, "files_regions": regions}
    return ds_definitions


# Aggregating function

def _aggregate_dbs(target, tmp_dbs_infos, mode="w"):
    """
    copy a list of .h5 files as defined in ``tmp_dbs_infos`` into ``target``

    Parameters
    ----------
    target : str
        name for the target file
    tmp_dbs_infos : list
        see the returned value of ``_make_db_for_each_file`` for the expected type
    mode : str
        the mode with which ``target`` is opened the first time. Must be 'w' if ``target`` doesn't exist.
    """
    features_infos = _aggregate_db_infos(tmp_dbs_infos)

    # open the file & create the master datasets to host the features
    with h5py.File(target, mode) as f:
        # add the list of features
        f.attrs["features"] = list(features_infos.keys())
        for name, params in features_infos.items():
            f.create_dataset(name, shape=params["shape"], dtype=params["dtype"],
                             chunks=True, maxshape=(None, *params["shape"][1:]))

    # prepare args for copying the tmp_files into the appropriate regions
    args = []
    for feature, info in features_infos.items():
        # collect the arguments for copying each file into its regions in the master datasets/features
        args += [(source, feature, indices) for source, indices in
                 zip(info["files_regions"].index, info["files_regions"].slices(time_axis=0))]
        # "<feature>_files" will store the slices corresponding to the files for this feature
        files_regions = info["files_regions"]
        files_regions.index = files_regions.index.str.replace(".tmp_", "").str.replace(".h5", "")
        files_regions = files_regions.reset_index(drop=False)
        files_regions = files_regions.rename(columns={"index": "name"})
        pd.DataFrame(files_regions).to_hdf(target, feature + "_files", mode="r+")

    # copy the data
    intra_regions = {}
    with h5py.File(target, "r+") as trgt:
        for source, key, indices in args:
            with h5py.File(source, "r") as src:
                data = src[key][()]
                attrs = {k: v for k, v in src[key].attrs.items()}
                if key + "_regions" in src:
                    if key not in intra_regions:
                        intra_regions[key] = []
                    # concat the regions :
                    regions = pd.read_hdf(source, key=key + "_regions", mode="r")
                    regions.loc[:, ("start", "stop")] += indices[0].start
                    regions = regions.reset_index(drop=False)
                    # remove ".tmp_" from the source's name
                    regions.loc[:, "name"] = "".join(source.split(".tmp_"))
                    intra_regions[key] += [regions]
            trgt[key][indices] = data
            trgt[key].attrs.update(attrs)
    # "<feature>_regions" will store the segmenting slices for this feature
    for key, regions in intra_regions.items():
        if regions is not None:
            pd.concat(regions).reset_index(drop=True).to_hdf(target, key + "_regions", mode="r+")

    # remove the temp_dbs
    for src, _ in tmp_dbs_infos:
        if src != target:
            os.remove(src)


def make_root_db(db_name,
                 files_iterable=tuple(),
                 extract_func=None,
                 tmp_destination="/tmp/",
                 n_cores=cpu_count()
                 ):
    """
    extract and aggregate several files into a single .h5 Database

    Parameters
    ----------
    db_name : str
        the name of the db to be created
    files_iterable : iterable of str, optional
        the files to be extracted
    extract_func : function, optional
        the function to use for the extraction - should take exactly one argument.
        the default transforms the file to the stft with n_fft=2048 and hop_length=512.
    tmp_destination : str, optional
    n_cores : int, optional
        the number of cores to use to parallelize the extraction process.
        default is the number of available cores on the system.

    Returns
    -------
    db : Database
        the created db
    """
    tmp_dbs_infos = _make_db_for_each_file(files_iterable, extract_func, tmp_destination, n_cores)
    _aggregate_dbs(db_name, tmp_dbs_infos, "w")
