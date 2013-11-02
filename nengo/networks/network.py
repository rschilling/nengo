from .. import objects

class Network(object):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.objects = []
        self.make(*args, **kwargs)

    def add(self, obj):
        self.objects.append(obj)
        return obj

    def make(self, *args, **kwargs):
        raise NotImplementedError("Networks should implement this function.")

    def add_to_model(self, model):
        for obj in self.objects:
            obj.name = self.name + '.' +  obj.name
            model.add(obj)
            
            # if we've added a Node, we need to also connect it to the 't'
            # signal so that it can change over time.
            if isinstance(obj, objects.Node):
                if len(obj.connections_in)==0:
                    model.connect(model.t, obj, filter=None)
            
