"""
Author: Brian O'Sullivan
Email: bmw.osullivan@gmail.com
Webpage: github.com/mo-geometry
Date: 08-10-2024
"""
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
import random
from scipy.spatial import Delaunay

SAVE_FRAMES = False


class TextBoxPanel(tk.Frame):
    def __init__(self, parent, title='angle1'):
        super().__init__(parent)

        # Calculate the width of the first text box based on the title length
        width1 = max(len(title) - 1, 3)  # Ensure a minimum width of 3

        # Create the first text box with dynamic width
        self.text_box1 = tk.Entry(self, width=width1)
        self.text_box1.insert(0, title)
        self.text_box1.grid(row=0, column=0, padx=(2, 2), pady=5)

        # Create the second text box with label '57'
        self.text_box2 = tk.Entry(self, width=5, justify='center')
        self.text_box2.insert(0, '57')
        self.text_box2.grid(row=0, column=1, padx=(1, 1), pady=5)


class QuaternionProductPanel(tk.Frame):
    def __init__(self, parent, colour='cyan'):
        super().__init__(parent, bg=colour)
        self.colour = colour
        # Title label
        self.title_label = tk.Label(self, text="Quaternion product", bg=colour)
        self.title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # TextBoxPanel instances
        self.beta_panel = TextBoxPanel(self, title='beta')
        self.beta_panel.text_box2.delete(0, tk.END)
        self.beta_panel.text_box2.insert(0, '-20')
        self.beta_panel.grid(row=1, column=0, padx=2, pady=2)

        self.theta_panel = TextBoxPanel(self, title='theta')
        self.theta_panel.text_box2.delete(0, tk.END)
        self.theta_panel.text_box2.insert(0, '30')
        self.theta_panel.grid(row=1, column=1, padx=2, pady=2)

        self.phi_panel = TextBoxPanel(self, title='phi')
        self.phi_panel.text_box2.delete(0, tk.END)
        self.phi_panel.text_box2.insert(0, '-150')
        self.phi_panel.grid(row=1, column=2, padx=2, pady=2)


class Q1(tk.Frame):
    def __init__(self, parent, title="Quaternion1", colour='lightblue', beta=5, theta=10, phi=20):
        super().__init__(parent, bg=colour)
        self.parent = parent
        self.title_label = tk.Label(self, text=title, bg='gray')
        self.title_label.pack(pady=10)
        self.colour = colour
        self.title = title
        self.init = False

        # Create IntVar for each angle
        self.beta_var = tk.IntVar()
        self.theta_var = tk.IntVar()
        self.phi_var = tk.IntVar()

        # Attach trace to call multiply_quaternions when values change
        self.beta_var.trace_add("write", self.on_value_change)
        self.theta_var.trace_add("write", self.on_value_change)
        self.phi_var.trace_add("write", self.on_value_change)

        # Create the sliders
        self.beta = tk.Scale(self, from_=-360, to=360, orient='horizontal', label="beta (°)", variable=self.beta_var)
        self.beta.pack(fill='x', padx=10, pady=5)
        self.beta.set(beta)

        self.theta = tk.Scale(self, from_=1, to=179, orient='horizontal', label="theta (°)", variable=self.theta_var)
        self.theta.pack(fill='x', padx=10, pady=5)
        self.theta.set(theta)

        self.phi = tk.Scale(self, from_=-180, to=180, orient='horizontal', label="phi (°)", variable=self.phi_var)
        self.phi.pack(fill='x', padx=10, pady=5)
        self.phi.set(phi)

        # initialization finished
        self.init = True

    def on_value_change(self, *args):
        self.update_values()
        if self.init:
            self.master.master.multiply_quaternions()

    def update_values(self):
        # extract angles
        beta = self.beta_var.get()
        theta = self.theta_var.get()
        phi = self.phi_var.get()

        # calculate quaternion values
        a, b, c, d = self.master.master.axis_angles_to_quaternion(beta, theta, phi)
        self.a, self.b, self.c, self.d = a, b, c, d

        # round for printing purposes
        # a1, b1, c1, d1 = np.round(a, 6), np.round(b, 6), np.round(c, 6), np.round(d, 6)
        # print(f"{self.title} ({self.colour}): [(a, b, c, d) = ({a1}, {b1}, {c1}, {d1})]")


class CheckBoxPanel(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # Title label
        self.title_label = tk.Label(self, text="Plot Options", font=("Arial", 14, "bold"))
        self.title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # First check box
        self.fill_enclosed_area_var = tk.BooleanVar()
        self.fill_enclosed_area = tk.Checkbutton(self, text='Fill enclosed area', variable=self.fill_enclosed_area_var)
        self.fill_enclosed_area.grid(row=1, column=0, sticky='w', padx=5, pady=5)

        # Second check box
        self.show_tangent_vectors_var = tk.BooleanVar()
        self.show_tangent_vectors = tk.Checkbutton(self, text='Show tangent vectors',
                                                   variable=self.show_tangent_vectors_var)
        self.show_tangent_vectors.grid(row=2, column=0, sticky='w', padx=5, pady=5)

        # Third check box
        self.show_3_vectors_var = tk.BooleanVar()
        self.show_3_vectors = tk.Checkbutton(self, text='Show 3-vectors', variable=self.show_3_vectors_var)
        self.show_3_vectors.grid(row=3, column=0, sticky='w', padx=5, pady=5)

        # Attach trace to variables
        self.fill_enclosed_area_var.trace_add("write", self.on_value_change)
        self.show_tangent_vectors_var.trace_add("write", self.on_value_change)
        self.show_3_vectors_var.trace_add("write", self.on_value_change)

    def on_value_change(self, *args):
        self.master.master.plot_sphere()


class CustomApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Custom Tkinter App")
        self.minsize(1250, 850)
        self.frame = 100
        self.initialize_sphere()

        self.left_panel = tk.Frame(self, width=350, bg='lightblue')
        self.left_panel.pack(side='left', fill='y')
        self.left_panel.pack_propagate(False)

        self.right_panel = tk.Frame(self, bg='lightgrey')
        self.right_panel.pack(side='right', fill='both', expand=True)

        self.checkbox_panel = CheckBoxPanel(self.left_panel)
        self.checkbox_panel.pack(fill='x', padx=10, pady=10)

        # Attach the Q1 panel to the left_panel
        self.q1_panel = Q1(self.left_panel, title='Quaternion1', colour='lightgreen', beta=151, theta=61, phi=-6)
        self.q1_panel.pack(fill='x', padx=10, pady=10)

        # Attach the Q1 panel to the left_panel
        self.q2_panel = Q1(self.left_panel, title='Quaternion2', colour='orange', beta=133, theta=25, phi=-37)
        self.q2_panel.pack(fill='x', padx=10, pady=10)

        # Attach the QuaternionProductPanel to the left_panel
        self.quaternion_product_panel = QuaternionProductPanel(self.left_panel, colour='magenta')
        self.quaternion_product_panel.pack(fill='x', padx=10, pady=10)

        # Create a save figure button
        self.save_button = tk.Button(self.left_panel, text='Save Figure', command=self.save_figure, width=10)
        self.save_button.pack(fill='x', padx=10, pady=10)

        self.canvas = None
        self.colormap = self.random_colormap()
        self.multiply_quaternions()
        self.plot_sphere()

    def save_figure(self):
        # open a directory dialog to save the figure, open in current directory
        file_path = tk.filedialog.asksaveasfilename(defaultextension='.png', initialdir='.')
        if file_path:
            self.canvas.figure.savefig(file_path, dpi=300)

    @staticmethod
    def quaternion_to_axis_angle(a, b, c, d):
        # ensure unit quaternion
        norm = np.sqrt(a ** 2 + b ** 2 + c ** 2 + d ** 2)
        a, b, c, d = a / norm, b / norm, c / norm, d / norm
        # calculate principle angle
        beta = np.arccos(a) * 2
        x, y, z = b / np.sin(beta / 2), c / np.sin(beta / 2), d / np.sin(beta / 2)
        # calculate 2-sphere angles
        theta = np.arccos(z)
        phi = np.arctan2(y, x)
        # return angles
        return np.degrees(beta), np.degrees(theta), np.degrees(phi)

    @staticmethod
    def axis_angles_to_quaternion(beta, theta, phi):
        a = np.cos(np.radians(beta) / 2)
        b = np.sin(np.radians(beta) / 2) * np.sin(np.radians(theta)) * np.cos(np.radians(phi))
        c = np.sin(np.radians(beta) / 2) * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
        d = np.sin(np.radians(beta) / 2) * np.cos(np.radians(theta))
        return a, b, c, d

    def quaternion_variables(self):
        # variables
        q1_theta, q1_phi = np.radians(self.q1_panel.theta_var.get()), np.radians(self.q1_panel.phi_var.get())
        q2_theta, q2_phi = np.radians(self.q2_panel.theta_var.get()), np.radians(self.q2_panel.phi_var.get())
        q1_beta, q2_beta = np.radians(self.q1_panel.beta_var.get()), np.radians(self.q2_panel.beta_var.get())
        q1 = {'n': [np.sin(q1_theta) * np.cos(q1_phi), np.sin(q1_theta) * np.sin(q1_phi), np.cos(q1_theta)],
              'beta': q1_beta, 'theta': q1_theta, 'phi': q1_phi, 'colour': self.q1_panel.colour}
        q2 = {'n': [np.sin(q2_theta) * np.cos(q2_phi), np.sin(q2_theta) * np.sin(q2_phi), np.cos(q2_theta)],
              'beta': q2_beta, 'theta': q2_theta, 'phi': q2_phi, 'colour': self.q2_panel.colour}
        # quaternion product
        q3_theta, q3_phi = np.radians(self.q3['theta']), np.radians(self.q3['phi'])
        q3_beta = np.radians(self.q3['beta'])
        q3 = {'n': [np.sin(q3_theta) * np.cos(q3_phi), np.sin(q3_theta) * np.sin(q3_phi), np.cos(q3_theta)],
              'beta': q3_beta, 'theta': q3_theta, 'phi': q3_phi, 'colour': self.quaternion_product_panel.colour}
        # quaternions
        return q1, q2, q3

    def plot_sphere(self, s=0.2, scale=0.985):
        # variables
        q1, q2, q3 = self.quaternion_variables()

        # Record the current viewpoint angles if the canvas exists
        if self.canvas:
            ax = self.canvas.figure.axes[0]
            elev, azim = ax.elev, ax.azim
            self.canvas.get_tk_widget().destroy()
        else:
            elev, azim = 11, 36

        fig = Figure()
        ax = fig.add_subplot(111, projection='3d')

        xS, yS, zS = scale * self.sphere['x'], scale * self.sphere['y'], scale * self.sphere['z']
        # Randomize the color map
        ax.plot_surface(xS, yS, zS, cmap=self.colormap, alpha=0.25)

        # Plot the x, y, z axes
        ax.quiver(0, 0, 0, 1, 0, 0, color='black', arrow_length_ratio=0.05, alpha=0.25)
        ax.quiver(0, 0, 0, 0, 1, 0, color='black', arrow_length_ratio=0.05, alpha=0.25)
        ax.quiver(0, 0, 0, 0, 0, 1, color='black', arrow_length_ratio=0.05, alpha=0.25)

        # Plot the 3-vectors
        if self.checkbox_panel.show_3_vectors_var.get():
            ax.quiver(0, 0, 0, q1['n'][0], q1['n'][1], q1['n'][2],
                      color=q1['colour'], arrow_length_ratio=0.05, linewidth=4, alpha=0.75)
            ax.quiver(0, 0, 0, q2['n'][0], q2['n'][1], q2['n'][2],
                      color=q2['colour'], arrow_length_ratio=0.05, linewidth=4, alpha=0.75)
            ax.quiver(0, 0, 0, q3['n'][0], q3['n'][1], q3['n'][2],
                      color=q3['colour'], arrow_length_ratio=0.05, linewidth=4, alpha=0.75)

        # Plot the bounded region with a different color
        if self.checkbox_panel.fill_enclosed_area_var.get():
            iX, iY, iZ = self.sphere['x'].flatten(), self.sphere['y'].flatten(), self.sphere['z'].flatten()
            ax.plot(iX[self.bounded], iY[self.bounded], iZ[self.bounded], 'co', alpha=0.25, markersize=6)

        # Plot the arcs between the vectors
        ax.plot(self.arc['AB'][:, 0], self.arc['AB'][:, 1], self.arc['AB'][:, 2],
                color='black', linewidth=4, alpha=0.95)
        ax.plot(self.arc['BC'][:, 0], self.arc['BC'][:, 1], self.arc['BC'][:, 2],
                color='black', linewidth=4, alpha=0.95)
        ax.plot(self.arc['CA'][:, 0], self.arc['CA'][:, 1], self.arc['CA'][:, 2],
                color='black', linewidth=4, alpha=0.95)

        # Plot the points of the 3-vectors
        for q in [q1, q2, q3]:
            ax.plot([q['n'][0]], [q['n'][1]], [q['n'][2]], q['colour'], marker='*', markersize=6)

        # Plot the tangent vectors
        if self.checkbox_panel.show_tangent_vectors_var.get():
            for X0, X1 in zip([q1['n'], q1['n'], q2['n'], q2['n'], q3['n'], q3['n']],
                              ['AB', 'AC', 'BA', 'BC', 'CA', 'CB']):
                x0, y0, z0 = X0[0], X0[1], X0[2]
                x1, y1, z1 = s * self.tangent[X1][0], s * self.tangent[X1][1], s * self.tangent[X1][2]
                # plot the tangent vectors
                # ax.plot([x0, x0 + x1], [y0, y0 + y1], [z0, z0 + z1], color='green', linewidth=2)
                ax.quiver(x0, y0, z0, x1, y1, z1,
                          color='cyan', arrow_length_ratio=0.35, linewidth=4, alpha=0.95, length=1.0)

        # Show the steradian measure in a text box
        ax.text2D(0.15, 0.85, f"Enclosed Area: {self.enclosed_area / np.pi:.2f} \u03C0 steradians",
                  transform=ax.transAxes)

        # Label the axes
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # Set specific tick labels
        ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        ticks_label = ['-1', '', '-0.5', '', '0', '', '0.5', '', '1']
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)
        ax.set_xticklabels(ticks_label)
        ax.set_yticklabels(ticks_label)
        ax.set_zticklabels(ticks_label)

        # Add grid lines
        ax.grid(True)

        # Restore the previous viewpoint angles if they exist
        if elev is not None and azim is not None:
            ax.view_init(elev=elev, azim=azim)
            # print('azi = ' + str(azim) + ' elev = ' + str(elev))

        # Adjust subplot parameters to minimize white space
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.canvas = FigureCanvasTkAgg(fig, master=self.right_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def multiply_quaternions(self):
        q1 = self.q1_panel
        q2 = self.q2_panel

        a = q1.a * q2.a - q1.b * q2.b - q1.c * q2.c - q1.d * q2.d
        b = q1.a * q2.b + q1.b * q2.a + q1.c * q2.d - q1.d * q2.c
        c = q1.a * q2.c - q1.b * q2.d + q1.c * q2.a + q1.d * q2.b
        d = q1.a * q2.d + q1.b * q2.c - q1.c * q2.b + q1.d * q2.a

        # save product for future use
        beta, theta, phi = self.quaternion_to_axis_angle(a, b, c, d)
        self.q3 = {'a': a, 'b': b, 'c': c, 'd': d, 'beta': beta, 'theta': theta, 'phi': phi}
        self.q2 = {'a': q2.a, 'b': q2.b, 'c': q2.c, 'd': q2.d, 'beta': self.q2_panel.beta_var.get(),
                   'theta': self.q2_panel.theta_var.get(), 'phi': self.q2_panel.phi_var.get()}
        self.q1 = {'a': q1.a, 'b': q1.b, 'c': q1.c, 'd': q1.d, 'beta': self.q1_panel.beta_var.get(),
                   'theta': self.q1_panel.theta_var.get(), 'phi': self.q1_panel.phi_var.get()}

        # variables
        q1_theta, q1_phi = np.radians(self.q1['theta']), np.radians(self.q1['phi'])
        q2_theta, q2_phi = np.radians(self.q2['theta']), np.radians(self.q2['phi'])
        q3_theta, q3_phi = np.radians(self.q3['theta']), np.radians(self.q3['phi'])
        q1_n = [np.sin(q1_theta) * np.cos(q1_phi), np.sin(q1_theta) * np.sin(q1_phi), np.cos(q1_theta)]
        q2_n = [np.sin(q2_theta) * np.cos(q2_phi), np.sin(q2_theta) * np.sin(q2_phi), np.cos(q2_theta)]
        q3_n = [np.sin(q3_theta) * np.cos(q3_phi), np.sin(q3_theta) * np.sin(q3_phi), np.cos(q3_theta)]

        # normal vectors
        self.q1['n'], self.q2['n'], self.q3['n'] = q1_n, q2_n, q3_n

        # round for printing purposes
        # a1, b1, c1, d1 = np.round(a, 6), np.round(b, 6), np.round(c, 6), np.round(d, 6)
        # print(f"Multiplied Quaternion: [(a, b, c, d) = ({a1}, {b1}, {c1}, {d1})]")

        # Update the QuaternionProductPanel with new values
        self.quaternion_product_panel.beta_panel.text_box2.delete(0, tk.END)
        self.quaternion_product_panel.beta_panel.text_box2.insert(0, f"{np.round(beta, 2):.2f}")
        self.quaternion_product_panel.theta_panel.text_box2.delete(0, tk.END)
        self.quaternion_product_panel.theta_panel.text_box2.insert(0, f"{np.round(theta, 2):.2f}")
        self.quaternion_product_panel.phi_panel.text_box2.delete(0, tk.END)
        self.quaternion_product_panel.phi_panel.text_box2.insert(0, f"{np.round(phi, 2):.2f}")

        # spherical triangle
        self.spherical_triangle()

        # bounded area
        self.bounded = self.winding_number_sphere()

        # enclosed area of the spherical triangle
        self.enclosed_area_spherical_triangle()

        # update the plot
        self.plot_sphere()

        # save frames
        if SAVE_FRAMES:
            self.save_frames()

    def save_frames(self):
        # save frames
        self.canvas.figure.savefig('frames/frame%03d.png' % self.frame, dpi=90)
        self.frame += 1

    def enclosed_area_spherical_triangle(self):
        ab, bc, ca = self.tangent['AB'], self.tangent['BC'], self.tangent['CA']
        ba, cb, ac = self.tangent['BA'], self.tangent['CB'], self.tangent['AC']
        # normalize
        ab, bc, ca = ab / np.linalg.norm(ab), bc / np.linalg.norm(bc), ca / np.linalg.norm(ca)
        ba, cb, ac = ba / np.linalg.norm(ba), cb / np.linalg.norm(cb), ac / np.linalg.norm(ac)
        # angles of the spherical triangle
        angle1 = np.arccos(np.clip(np.dot(ab, ac), -1.0, 1.0))
        angle2 = np.arccos(np.clip(np.dot(ba, bc), -1.0, 1.0))
        angle3 = np.arccos(np.clip(np.dot(ca, cb), -1.0, 1.0))
        # steradians
        self.enclosed_area = angle1 + angle2 + angle3 - np.pi

    def initialize_sphere(self, n_pts=200):
        u = np.linspace(0, 2 * np.pi, int(1.0 * n_pts))
        v = np.linspace(0, np.pi, n_pts)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        self.sphere = {'x': x, 'y': y, 'z': z, 'u': u, 'v': v}

    def winding_number_sphere(self):
        Xs, Ys = self.stereo_project_sphere()
        Px, Py = self.stereo_project_curve()
        dt = 1 / len(Px)
        # derivatives
        dPxdt, dPydt = np.gradient(Px) / dt, np.gradient(Py) / dt
        # point in curve
        point_in_curve = []
        for x, y in zip(Xs.flatten(), Ys.flatten()):
            A = dt * ((Px - x) * dPydt - (Py - y) * dPxdt) / ((Px - x) ** 2 + (Py - y) ** 2)
            point_in_curve.append(np.cumsum(A)[-2] / (2 * np.pi))
        bounded = np.abs(np.round(np.array(point_in_curve))) == 1
        return bounded

    def stereo_project_sphere(self):
        x, y, z = self.sphere['x'], self.sphere['y'], self.sphere['z']
        x, y, z = 0.5 * x, 0.5 * y, 0.5 * z + 0.5
        x, y, z = x, y, z + 1e-12
        # stereographic projection
        X = x / z
        Y = y / z
        return X, Y

    def stereo_project_curve(self):
        curve = np.concatenate([self.arc['AB'][:-1, :], self.arc['BC'][:-1, :], self.arc['CA']], axis=0)
        Rx, Ry, Rz = curve[:, 0], curve[:, 1], curve[:, 2]
        # stereographic projection
        x, y, z = 0.5 * Rx, 0.5 * Ry, 0.5 * Rz + 0.5
        x, y, z = x + 1e-12, y + 1e-12, z + 1e-12
        # stereographic projection
        X = x / z
        Y = y / z
        return X, Y

    def spherical_triangle(self):
        # vectors 3d
        A, B, C = self.q1['n'], self.q2['n'], self.q3['n']
        # arc between A and B
        arc_AB = self.arc_between_vectors(A, B)
        arc_BC = self.arc_between_vectors(B, C)
        arc_CA = self.arc_between_vectors(C, A)
        # assign to dictionary
        self.arc = {'AB': arc_AB, 'BC': arc_BC, 'CA': arc_CA}
        # tangent vectors
        tAB, tBA = self.return_tangent_vectors(A, B)
        tBC, tCB = self.return_tangent_vectors(B, C)
        tCA, tAC = self.return_tangent_vectors(C, A)
        # assign to dictionary
        self.tangent = {'AB': tAB, 'BA': tBA, 'BC': tBC, 'CB': tCB, 'CA': tCA, 'AC': tAC}

    @staticmethod
    def arc_between_vectors(v1, v2):
        # verify rotation of the quaternions
        theta_max = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        n = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
        # discrete steps in theta for the arc
        theta = np.linspace(0, theta_max, 101)
        # rotated vector B
        arc_x = v1[0] * np.cos(theta) + (n[1] * v1[2] - n[2] * v1[1]) * np.sin(theta)
        arc_y = v1[1] * np.cos(theta) + (n[2] * v1[0] - n[0] * v1[2]) * np.sin(theta)
        arc_z = v1[2] * np.cos(theta) + (n[0] * v1[1] - n[1] * v1[0]) * np.sin(theta)
        return np.array([arc_x, arc_y, arc_z]).T

    def spherical_triangle_test(self):
        # vectors 3d
        A, B, C = self.q1['n'], self.q2['n'], self.q3['n']
        # quaternions
        qAB = self.return_rotation_quaternion(A, B)
        qBC = self.return_rotation_quaternion(B, C)
        qCA = self.return_rotation_quaternion(C, A)
        # verify rotation of the quaternions
        B1 = self.quat_prod(self.quat_prod(qAB, [0, A[0], A[1], A[2]]), self.conj_quat(qAB))
        C1 = self.quat_prod(self.quat_prod(qBC, [0, B[0], B[1], B[2]]), self.conj_quat(qBC))
        A1 = self.quat_prod(self.quat_prod(qCA, [0, C[0], C[1], C[2]]), self.conj_quat(qCA))
        # calculate error
        if self.vec_err(A, A1[1:]) > 1e-6 or self.vec_err(B, B1[1:]) > 1e-6 > 1e-6 or self.vec_err(C, C1[1:]) > 1e-6:
            print('Error in quaternion rotation')
        # verify closed form
        self.verify_closed_form_quaternion_rotation(A, B)
        self.verify_closed_form_quaternion_rotation(B, C)
        self.verify_closed_form_quaternion_rotation(C, A)
        # get tangent vectors
        # self.return_tangent_vectors(A, B)

    @staticmethod
    def return_tangent_vectors(A, B):
        # verify rotation of the quaternions
        theta = np.arccos(np.clip(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B)), -1.0, 1.0))
        n = np.cross(A, B) / np.linalg.norm(np.cross(A, B))
        # rotated vector A to B
        # Bx = A[0] * np.cos(theta) + (n[1] * A[2] - n[2] * A[1]) * np.sin(theta)
        # By = A[1] * np.cos(theta) + (n[2] * A[0] - n[0] * A[2]) * np.sin(theta)
        # Bz = A[2] * np.cos(theta) + (n[0] * A[1] - n[1] * A[0]) * np.sin(theta)
        # Bvec = np.array([Bx, By, Bz])
        # derivative
        dBxdTHETA = -A[0] * np.sin(theta) + (n[1] * A[2] - n[2] * A[1]) * np.cos(theta)
        dBydTHETA = -A[1] * np.sin(theta) + (n[2] * A[0] - n[0] * A[2]) * np.cos(theta)
        dBzdTHETA = -A[2] * np.sin(theta) + (n[0] * A[1] - n[1] * A[0]) * np.cos(theta)
        # normalise
        normB = np.sqrt(dBxdTHETA ** 2 + dBydTHETA ** 2 + dBzdTHETA ** 2)
        # assign
        tBA = - np.array([dBxdTHETA, dBydTHETA, dBzdTHETA]) / normB
        # # rotated vector B to A
        # Ax = B[0] * np.cos(theta) - (n[1] * B[2] - n[2] * B[1]) * np.sin(theta)
        # Ay = B[1] * np.cos(theta) - (n[2] * B[0] - n[0] * B[2]) * np.sin(theta)
        # Az = B[2] * np.cos(theta) - (n[0] * B[1] - n[1] * B[0]) * np.sin(theta)
        # Avec = np.array([Ax, Ay, Az])
        # derivative
        dAxdTHETA = -B[0] * np.sin(theta) - (n[1] * B[2] - n[2] * B[1]) * np.cos(theta)
        dAydTHETA = -B[1] * np.sin(theta) - (n[2] * B[0] - n[0] * B[2]) * np.cos(theta)
        dAzdTHETA = -B[2] * np.sin(theta) - (n[0] * B[1] - n[1] * B[0]) * np.cos(theta)
        # normalise
        normA = np.sqrt(dAxdTHETA ** 2 + dAydTHETA ** 2 + dAzdTHETA ** 2)
        # assign
        tAB = - np.array([dAxdTHETA, dAydTHETA, dAzdTHETA]) / normA
        return tAB, tBA

    def verify_closed_form_quaternion_rotation(self, A, B):
        # verify rotation of the quaternions
        theta = np.arccos(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B)))
        n = np.cross(A, B) / np.linalg.norm(np.cross(A, B))
        # rotated vector A to B
        Bx = A[0] * np.cos(theta) + (n[1] * A[2] - n[2] * A[1]) * np.sin(theta)
        By = A[1] * np.cos(theta) + (n[2] * A[0] - n[0] * A[2]) * np.sin(theta)
        Bz = A[2] * np.cos(theta) + (n[0] * A[1] - n[1] * A[0]) * np.sin(theta)
        # verify
        if self.vec_err(B, [Bx, By, Bz]) > 1e-6:
            print('Error in closed form quaternion rotation')
        # rotated vector A to B
        Ax = B[0] * np.cos(theta) - (n[1] * B[2] - n[2] * B[1]) * np.sin(theta)
        Ay = B[1] * np.cos(theta) - (n[2] * B[0] - n[0] * B[2]) * np.sin(theta)
        Az = B[2] * np.cos(theta) - (n[0] * B[1] - n[1] * B[0]) * np.sin(theta)
        # verify
        if self.vec_err(A, [Ax, Ay, Az]) > 1e-6:
            print('Error in closed form quaternion rotation')

    @staticmethod
    def vec_err(A, B):
        return np.abs(np.array(A) - np.array(B)).sum()

    def conj_quat(self, q):
        # calculate the conjugate of a quaternion
        return [q[0], -q[1], -q[2], -q[3]]

    @staticmethod
    def return_rotation_quaternion(A, B):
        # calculate the rotation quaternion between two vectors
        n = np.cross(A, B) / np.linalg.norm(np.cross(A, B))
        theta = np.arccos(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B)))
        # calculate the quaternion
        a = np.cos(theta / 2)
        b = n[0] * np.sin(theta / 2)
        c = n[1] * np.sin(theta / 2)
        d = n[2] * np.sin(theta / 2)
        # verify norm
        norm = np.sqrt(a ** 2 + b ** 2 + c ** 2 + d ** 2)
        if norm != 1.0:
            # print('Norm of quaternion is not 1. Norm = %.12f' %norm)
            a, b, c, d = a / norm, b / norm, c / norm, d / norm
        return [a, b, c, d]

    @staticmethod
    def quat_prod(q1, q2):
        # calculate the product of two quaternions
        a = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        b = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
        c = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
        d = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
        return [a, b, c, d]

    @staticmethod
    def random_colormap(choice=True, colorcode=False, intense=False, pattern=False):
        options = [choice, colorcode, intense, pattern]
        if np.any(options) is not True:
            options[random.choice([0, 1, 2, 3])] = True
        if options[0]:  # choice
            x = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'pink', 'gray',
                 'gist_earth', 'gist_yarg', 'gist_gray', 'gist_heat', 'afmhot', 'ocean',
                 'cubehelix', 'binary', 'bone', 'copper', 'viridis', 'spring', 'summer',
                 'autumn', 'winter', 'cool', 'coolwarm', 'hot', 'plasma', 'inferno',
                 'magma', 'seismic', 'Wistia', 'Spectral', 'cividis']
        elif options[1]:  # colorcode
            x = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'YlOrBr', 'YlOrRd',
                 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', ]
        elif options[2]:  # intense
            x = ['gist_ncar', 'gist_rainbow', 'gist_stern', 'nipy_spectral', 'hsv', 'bwr',
                 'jet', 'rainbow', 'brg', 'terrain', 'gnuplot', 'gnuplot2', 'CMRmap', ]
        else:  # options[3]: pattern
            x = ['Pastel1', 'Pastel2', 'Dark2', 'Accent', 'Set1', 'Set2', 'Set3',
                 'flag', 'Paired', 'prism', 'tab10', 'tab20', 'tab20b', 'tab20c', ]
        return random.choice(x)

    # delunay triangulation
    @staticmethod
    def delaunay_triangulation(μ, ν):
        # surface
        i = np.ravel(np.sin(μ) * np.cos(ν))
        j = np.ravel(np.sin(μ) * np.sin(ν))
        k = np.ravel(np.cos(μ))
        # delunay triangulation
        return i, j, k, Triangulation(np.ravel(μ), np.ravel(ν))


if __name__ == "__main__":
    app = CustomApp()
    app.mainloop()
