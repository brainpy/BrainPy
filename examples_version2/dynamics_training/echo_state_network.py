# -*- coding: utf-8 -*-

import brainpy.version2 as bp
import brainpy.version2.math as bm

bm.set_environment(bm.batching_mode)


class ESN(bp.DynamicalSystem):
    def __init__(self, num_in, num_hidden, num_out):
        super(ESN, self).__init__()
        self.r = bp.dyn.Reservoir(num_in,
                                  num_hidden,
                                  Win_initializer=bp.init.Uniform(-0.1, 0.1),
                                  Wrec_initializer=bp.init.Normal(scale=0.1),
                                  in_connectivity=0.02,
                                  rec_connectivity=0.02,
                                  comp_type='dense')
        self.o = bp.dnn.Dense(num_hidden,
                              num_out,
                              W_initializer=bp.init.Normal(),
                              mode=bm.training_mode)

    def update(self, x):
        return x >> self.r >> self.o


class NGRC(bp.DynamicalSystem):
    def __init__(self, num_in, num_out):
        super(NGRC, self).__init__()

        self.r = bp.dyn.NVAR(num_in, delay=2, order=2)
        self.o = bp.dnn.Dense(self.r.num_out, num_out,
                              W_initializer=bp.init.Normal(0.1),
                              mode=bm.training_mode)

    def update(self, x):
        return x >> self.r >> self.o


def train_esn_with_ridge(num_in=100, num_out=30):
    model = ESN(num_in, 2000, num_out)

    # input-output
    print(model(bm.ones((1, num_in))))

    X = bm.random.random((1, 200, num_in))
    Y = bm.random.random((1, 200, num_out))

    # prediction
    runner = bp.DSTrainer(model, monitors=['r.state'])
    outputs = runner.predict(X)
    print(runner.mon['r.state'].shape)
    print(bp.losses.mean_absolute_error(outputs, Y))
    print()

    # training
    trainer = bp.RidgeTrainer(model)
    trainer.fit([X, Y])

    # prediction
    runner = bp.DSTrainer(model, monitors=['r.state'])
    outputs = runner.predict(X)
    print(runner.mon['r.state'].shape)
    print(bp.losses.mean_absolute_error(outputs, Y))
    print()

    outputs = trainer.predict(X)
    print(bp.losses.mean_absolute_error(outputs, Y))


def train_esn_with_force(num_in=100, num_out=30):
    model = ESN(num_in, 2000, num_out)

    # input-output
    print(model(bm.ones((1, num_in))))

    X = bm.random.random((1, 200, num_in))
    Y = bm.random.random((1, 200, num_out))

    # training
    trainer = bp.ForceTrainer(model, alpha=0.1)
    trainer.fit([X, Y])

    # prediction
    runner = bp.DSRunner(model, monitors=['r.state'])
    outputs = runner.predict(inputs=X)
    print(runner.mon['r.state'].shape)
    print(bp.losses.mean_absolute_error(outputs, Y))
    print()

    outputs = trainer.predict(X, reset_state=True)
    print(bp.losses.mean_absolute_error(outputs, Y))


def ngrc(num_in=10, num_out=30):
    model = NGRC(num_in, num_out)

    X = bm.random.random((1, 200, num_in))  # (num_batch, num_time, num_feature)
    Y = bm.random.random((1, 200, num_out))
    trainer = bp.RidgeTrainer(model, alpha=1e-6)
    outputs = trainer.predict(inputs=X)
    print(outputs.shape)
    print(bp.losses.mean_absolute_error(outputs, Y))
    trainer.fit([X, Y])
    outputs = trainer.predict(inputs=X)
    print(bp.losses.mean_absolute_error(outputs, Y))


def ngrc_bacth(num_in=10, num_out=30):
    model = NGRC(num_in, num_out)
    batch_size = 10
    model.reset_state(batch_size)
    X = bm.random.random((batch_size, 200, num_in))
    Y = bm.random.random((batch_size, 200, num_out))
    trainer = bp.RidgeTrainer(model, alpha=1e-6)
    outputs = trainer.predict(X)
    print(bp.losses.mean_absolute_error(outputs, Y))
    trainer.fit([X, Y])
    outputs = trainer.predict(X)
    print(bp.losses.mean_absolute_error(outputs, Y))


if __name__ == '__main__':
    train_esn_with_ridge(10, 30)
    train_esn_with_force(10, 30)
    ngrc(10, 30)
    ngrc_bacth()
