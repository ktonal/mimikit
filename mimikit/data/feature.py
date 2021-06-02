import dataclasses as dtc

__all__ = [
    'Feature'
]


@dtc.dataclass
class Feature:
    """
    class for centralizing methods pertaining to a particular type, class or format of data.
    """
    __ext__ = None

    @property
    def dim(self):
        """
        Returns
        -------
        dim: int, tuple
            information about the shape / dimensionality of the feature
        """
        raise NotImplementedError

    @property
    def params(self):
        """
        Returns
        -------
        params: dict
            the parameters of this Feature
        """
        return dtc.asdict(self)

    @property
    def encoders(self):
        """
        Returns
        -------
        encoders: dict
            dict or mapping where keys are types and values callables taking only one argument
        """
        class AnyIdentity:
            def __getitem__(self, item):
                return lambda x: x
        return AnyIdentity()

    @property
    def decoders(self):
        """
        Returns
        -------
        decoders: dict
            dict or mapping where keys are types and values callables taking only one argument
        """
        class AnyIdentity:
            def __getitem__(self, item):
                return lambda x: x

        return AnyIdentity()

    def __call__(self, inputs, encode=True):
        """
        only apply the layer of encoder/decoder owned by this class. Use it for augmentation, transformation.

        Parameters
        ----------
        inputs

        encode: bool, opt
            apply self.encoders if True, self.decoders otherwise

        Returns
        -------

        """
        funcs = self.encoders if encode else self.decoders
        return funcs[type(inputs)](inputs)

    def encode(self, inputs):
        """
        apply all the layers of encoders inherited by this class to `inputs`.

        Parameters
        ----------
        inputs

        Returns
        -------

        """
        if hasattr(super(), 'encoders'):
            inputs = super().encoders[type(inputs)](inputs)
        return self.encoders[type(inputs)](inputs)

    def decode(self, inputs):
        """
        apply all the layers of decoders inherited by this class to `inputs`.

        Parameters
        ----------
        inputs

        Returns
        -------

        """
        inputs = self.decoders[type(inputs)](inputs)
        if hasattr(super(), 'decoders'):
            inputs = super().decoders[type(inputs)](inputs)
        return inputs

    def load(self, path):
        """
        reads the file at `path` and returns it as numpy array

        Parameters
        ----------
        path

        Returns
        -------

        """
        # since this is optional, we don't raise any exception
        pass

    def post_create(self, db, schema_key):
        # since this is optional, we don't raise any exception
        pass

    def play(self, inputs):
        raise NotImplementedError

    def display(self, inputs):
        raise NotImplementedError

    def write(self, filename, inputs):
        raise NotImplementedError
