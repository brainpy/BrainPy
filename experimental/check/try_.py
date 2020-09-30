
# -*- coding: utf-8 -*-


class A():
    pars = None
    vars = None


def generate_class(pars_, vars_):
    class CreatedA(A):
        pars = pars_
        vars = vars_

    return CreatedA


# cls = generate_class([1, 2, 3], ['v', 'm', 'h'])
# print(cls)


from npbrain.core.neuron_group import create_neuron_model


print(create_neuron_model.__name__)

a = create_neuron_model(
    parameters={'a': 1, 'b': 2},
    variables=['m', 'v', 'h'],
    update_func=lambda a: a
)

print(a.vars)

