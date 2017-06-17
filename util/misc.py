import os
from six.moves import xrange

from IPython.display import SVG, display
import numpy as np
import svgwrite
import tensorflow as tf

from magenta.models.sketch_rnn.utils import get_bounds, to_big_strokes, to_normal_strokes
from magenta.models.sketch_rnn.model import sample


def draw_strokes(data, factor=0.2, svg_filename='/tmp/sketch_rnn/svg/sample.svg'):
    tf.gfile.MakeDirs(os.path.dirname(svg_filename))
    the_color, stroke_width = "black", 1
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in xrange(len(data)):
        command = 'm' if lift_pen == 1 else ('l' if command != 'l' else '')
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()
    display(SVG(dwg.tostring()))


def get_start_and_end(x):
    x = np.array(x)
    x = x[:, 0:2]
    x_start = x[0]
    x_end = x.sum(axis=0)
    x = x.cumsum(axis=0)
    x_max = x.max(axis=0)
    x_min = x.min(axis=0)
    center_loc = (x_max + x_min) * 0.5
    return x_start - center_loc, x_end


# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0):
    x_pos, y_pos = 0.0, 0.0
    result = [[x_pos, y_pos, 1]]

    for s_item in s_list:
        s, grid_loc = s_item
        grid_y = grid_loc[0] * grid_space + grid_space * 0.5
        grid_x = grid_loc[1] * grid_space_x + grid_space_x * 0.5
        start_loc, delta_pos = get_start_and_end(s)

        loc_x, loc_y = start_loc
        new_x_pos = grid_x + loc_x
        new_y_pos = grid_y + loc_y
        result.append([new_x_pos - x_pos, new_y_pos - y_pos, 0])

        result += s.tolist()
        result[-1][2] = 1
        x_pos = new_x_pos + delta_pos[0]
        y_pos = new_y_pos + delta_pos[1]

    return np.array(result)


# convenience functions to:
# 1) encode a stroke into a latent vector
# 2) decode from latent vector to stroke

class StrokeHelper(object):
    def __init__(self, eval_model, sample_model, sess):
        self.eval_model = eval_model
        self.sample_model = sample_model
        self.sess = sess

    def encode(self, input_strokes):
        strokes = to_big_strokes(input_strokes).tolist()
        strokes.insert(0, [0, 0, 1, 0, 0])

        draw_strokes(to_normal_strokes(np.array(strokes)))

        feed = {
            self.eval_model.input_data: [strokes],
            self.eval_model.sequence_lengths: [len(input_strokes)]
        }
        return self.sess.run(self.eval_model.batch_z, feed_dict=feed)[0]

    def decode(self, z_input=None, draw_mode=True, temperature=0.1, factor=0.2):
        z = [z_input] if z_input is not None else None
        sample_strokes, _ = sample(
            self.sess,
            self.sample_model,
            seq_len=self.eval_model.hps.max_seq_len,
            temperature=temperature,
            z=z
        )
        strokes = to_normal_strokes(sample_strokes)

        if draw_mode:
            draw_strokes(strokes, factor)

        return strokes
