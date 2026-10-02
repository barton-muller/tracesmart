import numpy as np
from PIL import Image

from tracesmart.render import render, shape_name


def test_shape_names():
    assert shape_name(3, None) == ("shape-3", "shape 3")
    assert shape_name(12, "window") == ("window-12", "window 12")
    assert shape_name(5, "dog's white chest") == ("dogs-white-chest-5", "dog's white chest 5")
    assert shape_name(1, "!!!")[0] == "described-1"


def test_render_names_described_shapes_and_colours_them():
    img = np.zeros((80, 80, 3), np.uint8)
    img[:] = (10, 120, 10)
    img[20:50, 20:60] = (200, 20, 20)
    background = np.ones((80, 80), bool)
    window = np.zeros((80, 80), bool)
    window[20:50, 20:60] = True
    svg, preview, segments = render(Image.fromarray(img), [background, window], [None, "window"], round_px=0)
    assert 'id="shape-0"' in svg and 'id="window-1"' in svg
    assert 'inkscape:label="window 1"' in svg
    assert 'fill="#c81414"' in svg  # the window's own colour
    assert 'stroke="#ff00ff"' in segments  # described shapes are outlined in the segment map
    assert preview.size == (80, 80)
    assert "L" in svg and "C" not in svg.split('id="window-1"')[1].split("/>")[0]  # a window stays rectangular
