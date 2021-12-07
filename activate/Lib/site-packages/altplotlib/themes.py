import matplotlib.pyplot as plt


def mpl_theme():
    line_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {
        "config": {
            "view": {"continuousWidth": 400, "continuousHeight": 300},
            "range": {"category": line_colors},
        }
    }
