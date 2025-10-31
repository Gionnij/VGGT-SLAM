class PickleAbleMixin:
    def __reduce__(self):
        return self.__class__, (self.__dict__,)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, obj):
        for k, v in obj.items():
            super(self.__class__, self).__setattr__(k, v)


class PointAccessMixin:
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError


class ConfigMap(dict, PointAccessMixin, PickleAbleMixin):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])

    This class was initially taken from
    https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary,
    then bugfixed and extended.
    To do, not inherit from map https://stackoverflow.com/questions/42320526/how-to-deal-with-pickle-load-calling-setitem-which-is-not-ready-for-use
    """

    def __init__(self, *args, **kwargs):
        super(ConfigMap, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    @staticmethod
    def recursive_create(value):
        if type(value) == dict:
            return ConfigMap(value)
        elif type(value) == list:
            return list(map(ConfigMap.recursive_create, value))
        else:
            return value

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        value_recursive = self.recursive_create(value)
        super(ConfigMap, self).__setitem__(key, value_recursive)
        self.__dict__.update({key: value_recursive})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(ConfigMap, self).__delitem__(key)
        del self.__dict__[key]

    def copy(self):
        return self.__class__(super(self.__class__, self).copy())

    def update(self, __m, **kwargs):
        for k, v in __m.items():
            self[k] = v
