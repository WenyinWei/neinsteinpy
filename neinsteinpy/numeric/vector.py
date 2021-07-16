from neinsteinpy.numeric.helpers import _change_name
from neinsteinpy.numeric.tensor import NBaseRelativityTensor, _change_config


class NGenericVector(NBaseRelativityTensor):
    """
    Class to represent a vector in arbitrary space-time numerically

    """

    def __init__(self, arr, var_arrs, config="u", parent_metric=None, name="GenericVector"):
        """
        Constructor and Initializer

        Parameters
        ----------
        arr : ~numpy.ndarray
            Numpy Array of the vector field 
        var_arrs : tuple or list of 1-dim numpy.ndarray
            Tuple of crucial symbols denoting time-axis, 1st, 2nd, and 3rd axis (t,x1,x2,x3)
        config : str
            Configuration of contravariant and covariant indices in tensor. 'u' for upper and 'l' for lower indices. Defaults to 'u'.
        parent_metric : ~neinsteinpy.numeric.metric.NMetricTensor or None
            Corresponding Metric for the Generic Vector.
            Defaults to None.
        name : str
            Name of the Vector. Defaults to "GenericVector".

        Raises
        ------
        ValueError
            config has more than 1 index
        ValueError
            Dimension should be equal to 1

        """
        super(NGenericVector, self).__init__(
            arr=arr, var_arrs=var_arrs, config=config, parent_metric=parent_metric, name=name
        )
        if len(self.arr.shape) - len(var_arrs) == 1:
            self._order = 1
            if not len(config) == self._order:
                raise ValueError("config should be of length {}".format(self._order))
        else:
            raise ValueError("Dimension should be equal to 1")

    def change_config(self, newconfig="u", metric=None):
        """
        Changes the index configuration(contravariant/covariant)

        Parameters
        ----------
        newconfig : str
            Specify the new configuration. Defaults to 'u'
        metric : ~neinsteinpy.numeric.metric.NMetricTensor or None
            Parent metric tensor for changing indices.
            Already assumes the value of the metric tensor from which it was initialized if passed with None.
            Defaults to None.

        Returns
        -------
        ~neinsteinpy.numeric.vector.NGenericVector
            New tensor with new configuration.

        Raises
        ------
        Exception
            Raised when a parent metric could not be found.

        """
        if metric is None:
            metric = self._parent_metric
        if metric is None:
            raise Exception("Parent Metric not found, can't do configuration change")
        new_tensor_arr = _change_config(self, metric, newconfig)
        new_obj = NGenericVector(
            new_tensor_arr,
            self.var_arrs,
            config=newconfig,
            parent_metric=metric,
            name=_change_name(self.name, context="__" + newconfig),
        )
        return new_obj

    def __add__(self, other):
        from numpy import allclose
        if self.arr.shape != other.arr.shape:
            raise ValueError("Mismatched shape of the two tensors. Please check they are expressed on the same variable mesh and ")
        if self.config != other.config:
            raise ValueError("Mismatch index config of the two tensors. Pleace check their index config are the same, e.g., 'uu', 'll'.")
        for i in range(len(self.var_arrs)):
            if not allclose(self.var_arrs[i], other.var_arrs[i]):
                raise ValueError("Mismatched variable mesh, you may consider interpolate one tensor on the other's tensor variable mesh.")
        return NGenericVector(
            self.arr + other.arr,
            self.var_arrs,
            config=self.config,
            parent_metric=self.parent_metric,
            name="GenericTensor",
        )
    def __neg__(self):
        return NGenericVector(
            -self.arr,
            self.var_arrs,
            config=self.config,
            parent_metric=self.parent_metric,
            name="GenericTensor",
        )
    def __sub__(self, other):
        return self + other.__neg__()

    # def lorentz_transform(self, transformation_matrix):
    #     """
    #     Performs a Lorentz transform on the vector.

    #     Parameters
    #     ----------
    #         transformation_matrix : ~sympy.tensor.array.dense_ndim_array.ImmutableDenseNDimArray or list
    #             Sympy Array or multi-dimensional list containing Sympy Expressions

    #     Returns
    #     -------
    #         ~einsteinpy.symbolic.vector.GenericVector
    #             lorentz transformed vector

    #     """

    #     t = super(GenericVector, self).lorentz_transform(transformation_matrix)
    #     return GenericVector(
    #         t.tensor(),
    #         var_arrs=self.var_arrs,
    #         config=self.config,
    #         parent_metric=None,
    #         name=_change_name(self.name, context="__lt"),
    #     )
