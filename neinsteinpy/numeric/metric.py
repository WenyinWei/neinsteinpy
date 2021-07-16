from neinsteinpy.numeric.helpers import _change_name
from neinsteinpy.numeric.tensor import NBaseRelativityTensor


class NMetricTensor(NBaseRelativityTensor):
    """
    Class to define a metric tensor for a space-time
    """

    def __init__(self, arr, var_arrs, config="ll", name="GenericNMetricTensor"):
        """
        Constructor and Initializer

        Parameters
        ----------
        arr : ~numpy.ndarray or list
            Numpy array of shape [dim, dim, time-space axis(like Nt,Nx1,Nx2,Nx3)]
        var_arrs : tuple or list of 1-dim numpy.ndarray
            Tuple of crucial symbols denoting time-axis, 1st, 2nd, and 3rd axis (t,x1,x2,x3)
        config : str
            Configuration of contravariant and covariant indices in tensor. 'u' for upper and 'l' for lower indices. Defaults to 'll'.
        name : str
            Name of the Metric Tensor. Defaults to "GenericNMetricTensor".

        Raises
        ------
        TypeError
            Raised when arr is not a list or numpy Array
        TypeError
            var_arrs is not a list or tuple of 1-dim numpy.ndarray
        ValueError
            config has more or less than 2 indices

        """
        super(NMetricTensor, self).__init__(
            arr=arr, var_arrs=var_arrs, config=config, parent_metric=self, name=name
        )
        self._order = 2
        self._invmetric = None
        if not len(config) == self._order:
            raise ValueError("config should be of length {}".format(self._order))

    def change_config(self, newconfig="uu"):
        """
        Changes the index configuration(contravariant/covariant)

        Parameters
        ----------
        newconfig : str
            Specify the new configuration. Defaults to 'uu'

        Returns
        -------
        ~neinsteinpy.numeric.metric.NMetricTensor
            New Metric with new configuration. Defaults to 'uu'

        Raises
        ------
        ValueError
            Raised when new configuration is not 'll' or 'uu'.
            This constraint is in place because we are dealing with Metric Tensor.

        """
        if newconfig == self.config:
            return self
        if newconfig == "uu" or newconfig == "ll":
            from numpy import moveaxis
            from numpy.linalg import inv
            inv_met = NMetricTensor(
                moveaxis(
                    inv( moveaxis(self.arr, [0, 1], [-2, -1])), [-2, -1], [0, 1] ),
                self.var_arrs,
                config=newconfig,
                name=_change_name(self.name, context="__" + newconfig),
            )
            inv_met._invmetric = self
            return inv_met

        raise ValueError(
            "Configuration can't have one upper and one lower index in Metric Tensor."
        )

    def inv(self):
        """
        Returns the inverse of the Metric.
        Returns contravariant Metric if it is originally covariant or vice-versa.

        Returns
        -------
        ~neinsteinpy.numeric.metric.NMetricTensor
            New Metric which is the inverse of original Metric.

        """
        if self._invmetric is None:
            if self.config == "ll":
                self._invmetric = self.change_config("uu")
            else:
                self._invmetric = self.change_config("ll")
        return self._invmetric

    def lower_config(self):
        """
        Returns a covariant instance of the given metric tensor.

        Returns
        -------
        ~neinsteinpy.numeric.metric.NMetricTensor
            same instance if the configuration is already lower or
            inverse of given metric if configuration is upper

        """
        if self.config == "ll":
            return self
        return self.inv()

    # def lorentz_transform(self, transformation_matrix):
    #     """
    #     Performs a Lorentz transform on the tensor.

    #     Parameters
    #     ----------
    #         transformation_matrix : ~sympy.tensor.array.dense_ndim_array.ImmutableDenseNDimArray or list
    #             Sympy Array or multi-dimensional list containing Sympy Expressions

    #     Returns
    #     -------
    #         ~einsteinpy.symbolic.metric.NMetricTensor
    #             lorentz transformed tensor

    #     """
    #     t = super(NMetricTensor, self).lorentz_transform(transformation_matrix)
    #     return NMetricTensor(
    #         t.tensor(),
    #         var_arrs=self.var_arrs,
    #         config=self._config,
    #         name=_change_name(self.name, context="__lt"),
    #     )
