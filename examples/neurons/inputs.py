import npbrain as nn

nn.profile.set_backend('numpy')
nn.profile.set_dt(0.02)

if __name__ == '__main__':
    neu = nn.FreqInput(100, freq=10)
    mon = nn.StateMonitor(neu, ['spike', 'spike_time'])
    net = nn.Network(neu=neu, mon=mon)
    net.run(duration=1001, report=True)

    ts = net.run_time()
    fig, gs = nn.vis.get_figure(1, 1, 5, 8)
    ax = fig.add_subplot(gs[0, 0])
    nn.vis.plot_raster(mon, ax=ax, show=True)
