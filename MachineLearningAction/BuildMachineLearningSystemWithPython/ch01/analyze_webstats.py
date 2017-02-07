import os
import scipy
import matplotlib.pyplot as plt
from utils import DATA_DIR, CHART_DIR

# set the seed for producing the random number
scipy.random.seed(3)

data = scipy.genfromtxt(os.path.join(DATA_DIR, "web_traffic.tsv"), delimiter="\t")
print(data)
print(data.shape)

colors = ['g', 'k', 'b', 'm', 'r']
linestyles = ['-', '-.', '--', ':', '-']


print(scipy.isnan(data))
print("The number of invalid entries is: ", scipy.sum(scipy.isnan(data)))

y = data[:, 1]
x = data[:, 0]
x = x[~scipy.isnan(y)]
y = y[~scipy.isnan(y)]

"""
def plot_models(x, y, models, fname, mx=None, ymax=None, xmin=None):
    plt.clf()
    plt.scatter(x, y, s=10)

    if models:
       if mx is None:
           mx = scipy.linspace(0, x[-1], num=1000)
           for model, style, color in zip(models, linestyles, colors):
               print(model(mx))
               plt.plot(mx. model(mx), linestyle=style)
"""

def plot_models(x, y, models, fname, mx=None, ymax=None, xmin=None):
    ''' plot input data '''

    plt.figure(num=None, figsize=(8, 6))
    plt.clf()
    plt.scatter(x, y, s=10)
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.xticks(
        [w * 7 * 24 for w in range(10)], ['week %i' % w for w in range(10)])

    if models:
        if mx is None:
            mx = scipy.linspace(0, x[-1], 1000)
        for model, style, color in zip(models, linestyles, colors):
            # print "Model:",model
            # print "Coeffs:",model.coeffs
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

    plt.autoscale(tight=True)
    plt.ylim(ymin=0)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)
    plt.grid(True, linestyle='-', color='0.75')
    if fname is not None:
        plt.savefig(fname)

# first have a look at the data
plot_models(x, y, None, None)

# create and plot models
fp1, res1, rank1, sv1, rcond1 = scipy.polyfit(x, y, 1, full=True)
print("Model parameters of fp1: %s" % fp1)
print("Error of the model of fp1: ", res1)
f1 = scipy.poly1d(fp1)

fp2, res2, rank2, sv2, rcond2 = scipy.polyfit(x, y, 2, full=True)
print("Model parameters of fp1: %s" % fp2)
print("Error of the model of fp1: ", res2)
f2 = scipy.poly1d(fp2)
f3 = scipy.poly1d(scipy.polyfit(x, y, 3))
f4 = scipy.poly1d(scipy.polyfit(x, y, 4))
f10 = scipy.poly1d(scipy.polyfit(x, y, 10))
f100 = scipy.poly1d(scipy.polyfit(x, y, 100))

plot_models(x, y, [f1], os.path.join(CHART_DIR, "1400_01_02.png"))
plot_models(x, y, [f1, f2], os.path.join(CHART_DIR, "1400_01_03.png"))
plot_models(
    x, y, [f1, f2, f3, f10, f100], os.path.join(CHART_DIR, "1400_01_04.png"))

inflection = 3.5 * 7 * 24
xa = x[:inflection]
ya = y[:inflection]
xb = x[inflection:]
yb = y[inflection:]

fa = scipy.poly1d(scipy.polyfit(xa, ya, deg=1))
fb = scipy.poly1d(scipy.polyfit(xb, yb, deg=1))

plot_models(x, y, [fa, fb], os.path.join(CHART_DIR, "1400_01_05.png"))

def error(f, x, y):
    return scipy.sum((f(x) - y) ** 2)

print("Errors for the complete data set:")
for f in [f1, f2, f3, f4, f10, f100]:
    print("Error d=%i, %f" % (f.order, error(f, x, y)))

print("Errors for only the time before the inflection point")
for f in [f1, f2, f3, f4, f10, f100]:
    print("Error d=%i, %f" % (f.order, error(f, xa, ya)))

print("Errors for only the time after the inflection point")
for f in [f1, f2, f3, f4, f10, f100]:
    print("Error d=%i, %f" % (f.order, error(f, xb, yb)))

plot_models(
    x, y, [f1, f2, f3, f10, f100],
    os.path.join(CHART_DIR, "1400_01_06.png"),
    mx=scipy.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)

# separating training data from testing data
frac = 0.3
split_idx = int(frac * len(xb))
shuffled = scipy.random.permutation(list(range(len(xb))))
# use the first part to be test data
test = sorted(shuffled[:split_idx])
# use the second part to be training data
train = sorted(shuffled[split_idx:])
fbt1 = scipy.poly1d(scipy.polyfit(xb[train], yb[train], 1))
fbt2 = scipy.poly1d(scipy.polyfit(xb[train], yb[train], 2))
fbt3 = scipy.poly1d(scipy.polyfit(xb[train], yb[train], 3))
fbt10 = scipy.poly1d(scipy.polyfit(xb[train], yb[train], 10))
fbt100 = scipy.poly1d(scipy.polyfit(xb[train], yb[train], 100))

for f in [fbt1, fbt2, fbt3, fbt10, fbt100]:
    print("Error d=%i: %f" % (f.order, error(f, xb[test], yb[test])))

plot_models(
    x, y, [fbt1, fbt2, fbt3, fbt10, fbt100],
    os.path.join(CHART_DIR, "1400_01_08.png"),
    mx=scipy.linspace(0 * 7 * 24, 6 * 7 * 24, 100),
    ymax=10000, xmin=0 * 7 * 24)

from scipy.optimize import fsolve
print(fbt2)
print(fbt2 - 100000)
reached_max = fsolve(fbt2 - 100000, x0=800) / (7 * 24)
print("100,000 hits.hour expected at week %f" % reached_max[0])