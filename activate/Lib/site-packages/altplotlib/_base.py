import altair as alt
import numpy as np
from altplotlib.data import df_prep
import matplotlib


class AltairFigure:
    def __init__(self):
        self.axes = []

    def combine(self):
        rows = []
        for row in self.axes:
            r = row[0]
            for col in row[1:]:
                r |= col.chart
            rows.append(r)

        c = rows[0]
        for row in rows[1:]:
            c &= row
        return c


class AltairAxis:
    coercible = (alt.Chart, alt.HConcatChart, alt.VConcatChart, alt.LayerChart)

    def __init__(self):
        self.chart = alt.LayerChart()

    def make_chart(self, x, y, labels):
        data = df_prep(x, y, labels)
        chart = alt.Chart(data).encode(
            x="x",
            y="y",
        )
        return chart

    def plot(self, x, y=None, labels=None, legend=None, c=None):
        chart = self.make_chart(x, y, labels)

        line_options = {}
        encode_options = {"detail": "series"}
        if c:
            line_options["color"] = matplotlib.colors.to_hex(c)
        else:
            encode_options["color"] = alt.Color("series:N", legend=legend)

        line = chart.mark_line(**line_options).encode(**encode_options)
        self.chart += line
        return self.chart

    def scatter(self, x, y, labels=None, legend=None, c=None):
        chart = self.make_chart(x, y, labels)

        line_options = {}
        encode_options = {"detail": "series"}
        if c:
            line_options["color"] = matplotlib.colors.to_hex(c)
        else:
            encode_options["color"] = alt.Color("series:N", legend=legend)

        point = chart.mark_circle(**line_options).encode(**encode_options)
        self.chart += point
        return self.chart

    def axhline(self, y):
        hline = alt.Chart(pd.DataFrame({"y": [y]})).mark_rule().encode(y="y")
        self.chart += hline
        return self.chart

    def axvline(self, x):
        vline = alt.Chart(pd.DataFrame({"x": [x]})).mark_rule().encode(x="x")
        self.chart += vline
        return self.chart

    def set_title(self, title):
        self.chart = self.chart.properties(title=str(title))
        return self.chart

    def set_xlabel(self, title):
        self.chart.layer[0].encoding.x.title = str(title)
        return self.chart

    def set_ylabel(self, title):
        self.chart.layer[0].encoding.y.title = str(title)
        return self.chart

    def __call__(self):
        return self.chart

    def __or__(self, other):
        if isinstance(other, self.__class__):
            return self.chart | other.chart
        if isinstance(other, self.coercible):
            return self.chart | other
        raise TypeError(
            f"unsupported operand type(s) for |: '{self.__class__.__name__}' and '{other.__class__.__name__}'"
        )

    def __ror__(self, other):
        if isinstance(other, self.coercible):
            return other | self.chart
        raise TypeError(
            f"unsupported operand type(s) for |: '{self.__class__.__name__}' and '{other.__class__.__name__}'"
        )

    def __and__(self, other):
        if isinstance(other, self.__class__):
            return self.chart & other.chart
        if isinstance(other, self.coercible):
            return self.chart & other
        raise TypeError(
            f"unsupported operand type(s) for &: '{self.__class__.__name__}' and '{other.__class__.__name__}'"
        )

    def __rand__(self, other):
        if isinstance(other, self.coercible):
            return other & self.chart
        raise TypeError(
            f"unsupported operand type(s) for &: '{self.__class__.__name__}' and '{other.__class__.__name__}'"
        )


def subplots(nrows, ncols):
    figure = AltairFigure()
    for r in range(nrows):
        row = []
        for c in range(ncols):
            row.append(AltairAxis())
        figure.axes.append(row)
    return figure, np.array(figure.axes)


def plot(*args, **kwargs):
    axis = AltairAxis()
    axis.plot(*args, **kwargs)
    return axis
