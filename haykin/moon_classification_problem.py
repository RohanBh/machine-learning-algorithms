from scipy import stats
import numpy as np


def moon_plot(ax, sample):
    (xa, ya), (xb, yb) = sample
    ax.scatter(x=xa, y=ya, marker='x')
    ax.scatter(x=xb, y=yb, marker='x')
    ax.grid(True)
    ax.axis('equal')
    return ax


class MoonProblem:
    class RadiusRandomVariable(stats.rv_continuous):
        # noinspection PyMethodOverriding
        def _pdf(self, x, r, w):
            if abs(x - r) > w: return 0
            return x / (2 * r * w)

        # noinspection PyMethodOverriding
        def _ppf(self, q, r, w):
            return np.sqrt(4 * r * w * q + (r - w) ** 2)

    def __init__(self, d=0, r=10, w=6):
        self.d = d
        self.r = r
        self.w = w / 2
        self.training_sample = None
        self.test_sample = None

    def _generate_data_points(self, num_pairs):
        """ Each data point consists of a pair of points picked from region A and region B randomly.
                :param num_pairs: Number of datapoints to generate. Default is 1000.
                """

        radius_rv = MoonProblem.RadiusRandomVariable()

        def generate():
            radius = radius_rv.rvs(self.r, self.w, size=num_pairs)
            angle = np.random.uniform(size=num_pairs) * np.pi
            return radius, angle

        def transform(r, theta): return r * np.cos(theta), r * np.sin(theta)

        xa, ya = transform(*generate())
        xb, yb = transform(*generate())
        xb, yb = xb + self.r, -yb - self.d
        return (xa, ya), (xb, yb)

    def get_training_sample(self, num_pairs=1000):
        if self.training_sample is not None: return self.training_sample
        self.training_sample = self._generate_data_points(num_pairs)
        return self.training_sample

    def get_test_sample(self, num_pairs=2000):
        if self.test_sample is not None: return self.test_sample
        self.test_sample = self._generate_data_points(num_pairs)
        return self.test_sample
