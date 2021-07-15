import numpy as np

from neinsteinpy.numeric.helpers import _change_name


def _config_checker(config):
    # check if the string for config contains 'u' and 'l' only
    if not isinstance(config, str):
        return False
    for ch in config:
        if (not ch == "l") and (not ch == "u"):
            return False
    return True


def _difference_list(newconfig, oldconfig):
    # defines a list of actions to be taken on a tensor
    difflist = list()
    for n_ch, o_ch in zip(newconfig, oldconfig):
        if n_ch == o_ch:
            difflist.append(0)
        elif n_ch == "u":
            difflist.append(1)
        else:
            difflist.append(-1)
    return difflist


def _change_config(tensor, metric, newconfig):
    # check length and validity of new configuration
    if not (len(newconfig) == len(tensor.config) and _config_checker(newconfig)):
        raise ValueError

    # seperate the contravariant & covariant metric tensors
    met_dict = {
        -1: metric.lower_config().tensor(),
        1: metric.lower_config().inv().tensor(),
    }

    # main code
    def chain_config_change():
        t = sympy.Array(tensor.tensor())
        difflist = _difference_list(newconfig, tensor.config)
        for i, action in enumerate(difflist):
            if action == 0:
                continue
            else:
                t = tensorcontraction(tensorproduct(met_dict[action], t), (1, 2 + i))
                # reshuffle the indices
                dest = list(range(len(t.shape)))
                dest.remove(0)
                dest.insert(i, 0)
                t = sympy.permutedims(t, dest)
        return t

    return chain_config_change()


def tensor_product(tensor1, tensor2, i=None, j=None):
    """Tensor Product of ``tensor1`` and ``tensor2``

    Parameters
    ----------
    tensor1 : ~neinsteinpy.numeric.NBaseRelativityTensor
    tensor2 : ~neinsteinpy.numeric.NBaseRelativityTensor
    i : int, optional
        contract ``i``th index of ``tensor1``
    j : int, optional
        contract ``j``th index of ``tensor2``


    Returns
    -------
    ~neinsteinpy.numeric.NBaseRelativityTensor
        tensor of appropriate rank

    Raises
    ------
    ValueError
        Raised when ``i`` and ``j`` both indicate 'u' or 'l' indices
    """
    import itertools 
    if (i or j) is None:
        newconfig = tensor1.config + tensor2.config

        if tensor1.order == 0 == tensor2.order:
            product = tensor1.arr * tensor2.arr
        else:
            dim = tensor1.arr.shape[0] if tensor1.order >= 1 else tensor2.arr.shape[0] 
            
            product = np.empty(
                (*tensor1.arr.shape[:tensor1.order],
                 *tensor2.arr.shape[:tensor2.order],
                 *tensor1.arr.shape[tensor1.order:]), dtype=tensor1.arr.dtype)
            for ind1 in list(itertools.product(range(dim), repeat=tensor1.order)):
                for ind2 in list(itertools.product(range(dim), repeat=tensor2.order)):
                    product[ind1+ind2] = tensor1.arr[ind1] * tensor2.arr[ind2]
    else:
        if tensor1.config[i] == tensor2.config[j]:
            raise ValueError(
                "Index summation not allowed between %s and %s indices"
                % (tensor1.config[i], tensor2.config[j])
            )

        tensor1_noncoord_shape = tensor1.arr.shape[:tensor1.order]
        tensor2_noncoord_shape = tensor2.arr.shape[:tensor2.order]
        product = np.zeros(
            (*tensor1_noncoord_shape[:i], *tensor1_noncoord_shape[i+1:],
             *tensor2_noncoord_shape[:j], *tensor2_noncoord_shape[j+1:],
             *tensor1.arr.shape[tensor1.order:]), dtype=tensor1.arr.dtype)
        
        for ind1 in list(itertools.product(range(tensor1.arr.shape[0]), repeat=tensor1.order-1)):
            for ind2 in list(itertools.product(range(tensor2.arr.shape[0]), repeat=tensor2.order-1)):
                for k in range(tensor1.arr.shape[i]):
                    ind1_iter = ind1[:i] + (k,) + ind1[i:]
                    ind2_iter = ind2[:j] + (k,) + ind2[j:]
                    product[ind1+ind2] += tensor1.arr[ind1_iter] * tensor2.arr[ind2_iter]

        con = tensor1.config[:i] + tensor1.config[i + 1 :]
        fig = tensor2.config[:j] + tensor2.config[j + 1 :]
        newconfig = con + fig

    return NBaseRelativityTensor(
        product,
        var_arrs=tensor1.var_arrs,
        config=newconfig,
        parent_metric=tensor1.parent_metric
    )


class NTensor:
    """
    Base Class for numeric Tensor manipulation
    """

    def __init__(self, arr, config="ll", name=None):
        """
        Constructor and Initializer

        Parameters
        ----------
        arr : ~numpy.ndarray or list
            Numpy Array, multi-dimensional list containing Sympy Expressions, or Sympy Expressions or int or float scalar
        config : str
            Configuration of contravariant and covariant indices in tensor. 'u' for upper and 'l' for lower indices. Defaults to 'll'.
        name : str or None
            Name of the tensor.

        Raises
        ------
        TypeError
            Raised when arr is not a list or Numpy array
        TypeError
            Raised when config is not of type str or contains characters other than 'l' or 'u'

        """

        if isinstance(arr, (list, tuple, int, float, np.number)):
            self.arr = np.array(arr)
        elif isinstance(arr, np.ndarray):
            self.arr = arr
        else:
            raise TypeError("Only multi-dimensional list or Numpy Array is expected")
        if _config_checker(config):
            self._config = config
            self._order = len(config)
        else:
            raise TypeError(
                "config is either not of type 'str' or does contain characters other than 'l' or 'u'"
            )
            
        self.name = name

    @property
    def order(self):
        """
        Returns the order of the Tensor

        """
        return self._order

    @property
    def config(self):
        """
        Returns the configuration of covariant and contravariant indices

        """
        return self._config

    def __getitem__(self, index):
        return self.arr[index]

    def __str__(self):
        """
        Returns a String with a readable representation of the object of class Tensor

        """
        representation = "Tensor"
        if self.name is not None:
            representation = " ".join((representation, self.name))
        representation += "\n"
        representation += self.arr.__str__() # TODO: is this necessary for numeric einsteinpy?
        return representation

    # def __repr__(self):
    #     """
    #     Returns a String with a representation of the state of the object of class Tensor

    #     """
    #     interpretable_representation = self.__class__.__name__
    #     interpretable_representation += self.arr.__repr__()
    #     return interpretable_representation

    def tensor(self):
        """
        Returns the numpy Array

        Returns
        -------
        ~numpy.ndarray
            Numpy Array object

        """
        return self.arr


class NBaseRelativityTensor(NTensor):
    """
    Generic class for defining tensors in General Relativity.
    This would act as a base class for other Tensorial quantities in GR.

    Attributes
    ----------
    arr : ~numpy.ndarray
        Raw Tensor in sympy array
    var_arrs : list or tuple of 1-dim ndarray 
        List of symbols denoting space and time axis
    dims : int
        dimension of the space-time.
    name : str or None
        Name of the tensor. Defaults to "GenericTensor".

    """

    def __init__(
        self,
        arr,
        var_arrs,
        config="ll",
        parent_metric=None,
        name="GenericTensor",
    ):
        """
        Constructor and Initializer

        Parameters
        ----------
        arr : ~numpy.ndarray or list
            Numpy Array or multi-dimensional list containing Sympy Expressions
        var_arrs : tuple or list of numpy.ndarray
            List of crucial variables dentoting time-axis and/or spacial axis.
            For example, in case of 4D space-time, the arrangement would look like [t, x1, x2, x3].
        config : str
            Configuration of contravariant and covariant indices in tensor.
            'u' for upper and 'l' for lower indices. Defaults to 'll'.
        parent_metric : ~neinsteinpy.numeric.metric.NMetricTensor or None
            Metric Tensor for some particular space-time which is associated with this Tensor.
        name : str or None
            Name of the Tensor. Defaults to "GenericTensor".

        Raises
        ------
        TypeError
            Raised when arr is not a list or numpy array.
        TypeError
            Raised when config is not of type str or contains characters other than 'l' or 'u'
        TypeError
            Raised when arguments var_arrs have data type other than list, tuple or set.
        TypeError
            Raised when argument parent_metric does not belong to NMetricTensor class and isn't None.
        ValueError
            Raised when argument ``var_arrs`` does not agree with shape of argument ``arr``

        """
        super(NBaseRelativityTensor, self).__init__(arr=arr, config=config, name=name)

        if len(self.arr.shape) != len(config) + len(var_arrs):
            raise ValueError("Mismatched shape of argument arr, config and var_arrs. To make the argument legal, the dimension of arr must equal len(config) + the number of space-time axis.")

        # Cannot implement the check that parent metric belongs to the class MetricTensor
        # Due to the issue of cyclic imports, would find a workaround
        self._parent_metric = parent_metric
        if isinstance(var_arrs, (list, tuple)):
            self.var_arrs = var_arrs
            self.dims = len(self.var_arrs)
        else:
            raise TypeError("var_arrs should be a list or tuple")

    @property
    def parent_metric(self):
        """
        Returns the Metric from which Tensor was derived/associated, if available.
        """
        return self._parent_metric

    def __add__(self, other):
        from numpy import allclose
        if self.arr.shape != other.arr.shape:
            raise ValueError("Mismatched shape of the two tensors. Please check they are expressed on the same variable mesh and ")
        if self.config != other.config:
            raise ValueError("Mismatch index config of the two tensors. Pleace check their index config are the same, e.g., 'uu', 'll'.")
        for i in range(len(self.var_arrs)):
            if not allclose(self.var_arrs[i], other.var_arrs[i]):
                raise ValueError("Mismatched variable mesh, you may consider interpolate one tensor on the other's tensor variable mesh.")
        return NBaseRelativityTensor(
            self.arr + other.arr,
            self.var_arrs,
            config=self.config,
            parent_metric=self.parent_metric,
            name="GenericTensor",
        )
    def __neg__(self):
        return NBaseRelativityTensor(
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
    #     Performs a Lorentz transform on the tensor.

    #     Parameters
    #     ----------
    #         transformation_matrix : ~sympy.tensor.array.dense_ndim_array.ImmutableDenseNDimArray or list
    #             Sympy Array or multi-dimensional list containing Sympy Expressions

    #     Returns
    #     -------
    #         ~einsteinpy.symbolic.tensor.BaseRelativityTensor
    #             lorentz transformed tensor(or vector)

    #     """
    #     tm = sympy.Array(transformation_matrix)
    #     t = self.tensor()
    #     for i in range(self.order):
    #         if self.config[i] == "u":
    #             t = tensorcontraction(tensorproduct(tm, t), (1, 2 + i))
    #         else:
    #             t = tensorcontraction(tensorproduct(tm, t), (0, 2 + i))
    #         dest = list(range(len(t.shape)))
    #         dest.remove(0)
    #         dest.insert(i, 0)
    #         t = sympy.permutedims(t, dest)

    #     return NBaseRelativityTensor(
    #         t,
    #         var_arrs=self.var_arrs,
    #         config=self.config,
    #         parent_metric=None,
    #         variables=self.variables,
    #         functions=self.functions,
    #         name=_change_name(self.name, context="__lt"),
    #     )
