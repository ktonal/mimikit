def demo():
    """### load a file and segment it"""
    import mimikit as mmk
    import IPython.display as ipd
    from ipywidgets import widgets as W

    ipd.display(mmk.MMK_STYLE_SHEET)
    ipd.display(W.HTML(
        """
        <style>
        .container {
            width: 95% !important;
        }
        """
    ))
    sr = 22050
    y = mmk.FileToSignal(sr=sr, duration=None)("./my-file.m4a")
    
    # the more overlap -> the more precise in time
    # the more grad_lag -> the smoother -> less attack/decay

    samplifyer = mmk.Samplifyer(
        levels_def=[
            # dict(n_fft=8192, overlap=16, grad_max_lag=5),
            # dict(n_fft=4096, overlap=4, grad_max_lag=13),
            dict(n_fft=2048, overlap=8, grad_max_lag=15),
            dict(n_fft=1024, overlap=8, grad_max_lag=9),
            dict(n_fft=512, overlap=8, grad_max_lag=7),
            dict(n_fft=256, overlap=4, grad_max_lag=7),
            # dict(n_fft=128, overlap=4, grad_max_lag=17),
        ]
    )
    samplifyer.fit(y)

    """### filter cuts based on their scores"""

    mmk.segment_selector_view(y, samplifyer.cuts, samplifyer.scores, sr)

    """### Note"""
    """
you can interact with the waveform with following shortcuts:

- Navigation:
    * `Ctrl + wheel`: zoom
    * `SHIFT + dbl-click`: reset zoom
    * `SHIFT + wheel`: scroll waveform
    * `arrow left/right`: move playhead left/right
    * `SHIFT + arrow left/right`: move playhead left/right a lot.
- Controls:
    * `dbl-click`: play from there
    * `SPACE BAR`: play/pause 
    """
    """----------------------------"""