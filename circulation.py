import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Circulation:
	"""
	Model of systemic circulation from Ferreira et al. (2005), A Nonlinear State-Space Model
	of a Combined Cardiovascular System and a Rotary Pump, IEEE Conference on Decision and Control.
	"""

	def __init__(self, HR, Emax, Emin):
		self.set_heart_rate(HR)

		self.Emin = Emin
		self.Emax = Emax
		self.non_slack_blood_volume = 250  # ml

		self.R1 = 1.0  # between .5 and 2
		self.R2 = .005
		self.R3 = .001
		self.R4 = .0398

		self.C2 = 4.4
		self.C3 = 1.33

		self.L = .0005

	def set_heart_rate(self, HR):
		"""
		Sets several related variables together to ensure that they are consistent.

		:param HR: heart rate (beats per minute)
		"""
		self.HR = HR
		self.tc = 60 / HR
		self.Tmax = .2 + .15 * self.tc  # contraction time

	def get_derivative(self, t, x):
		"""
		:param t: time (s)
		:param x: state variables [ventricular pressure; atrial pressure; arterial pressure; aortic flow]
		:return: time derivatives of state variables
		"""
		if x[1] > x[0]:
			return np.matmul(self.filling_phase_dynamic_matrix(t), x)

		if (x[3] > 0) or (x[0] > x[2]):
			return np.matmul(self.ejection_phase_dynamic_matrix(t), x)

		return np.matmul(self.isovolumic_phase_dynamic_matrix(t), x)

	def isovolumic_phase_dynamic_matrix(self, t):
		"""
		:param t: time (s; needed because elastance is a function of time)
		:return: A matrix for isovolumic phase
		"""
		el = self.elastance(t)
		del_dt = self.elastance_finite_difference(t)
		return [[del_dt / el, 0, 0, 0],
				[0, -1 / (self.R1 * self.C2), 1 / (self.R1 * self.C2), 0],
				[0, 1 / (self.R1 * self.C3), -1 / (self.R1 * self.C3), 0],
				[0, 0, 0, 0]]

	def ejection_phase_dynamic_matrix(self, t):
		"""
		:param t: time (s)
		:return: A matrix for filling phase
		"""
		el = self.elastance(t)
		del_dt = self.elastance_finite_difference(t)
		return [[del_dt / el, 0, 0, -el],
				[0, -1 / (self.R1 * self.C2), 1 / (self.R1 * self.C2), 0],
				[0, 1 / (self.R1 * self.C3), -1 / (self.R1 * self.C3), 1 / self.C3],
				[1 / self.L, 0, -1 / self.L, -(self.R3 + self.R4) / self.L]]

	def filling_phase_dynamic_matrix(self, t):
		"""
		:param t: time (s)
		:return: A matrix for filling phase
		"""
		el = self.elastance(t)
		del_dt = self.elastance_finite_difference(t)
		return [[del_dt / el - el / self.R2, el / self.R2, 0, 0],
				[1 / (self.R2 * self.C2), -(self.R1 + self.R2) / (self.R1 * self.R2 * self.C2), 1 / (self.R1 * self.C2),
				 0],
				[0, 1 / (self.R1 * self.C3), -1 / (self.R1 * self.C3), 0],
				[0, 0, 0, 0]]

	def elastance(self, t):
		"""
		:param t: time (s; needed because elastance is a function of time)
		:return: time-varying elastance
		"""
		tn = self._get_normalized_time(t)
		En = 1.55 * np.power(tn / .7, 1.9) / (1 + np.power(tn / .7, 1.9)) / (1 + np.power(tn / 1.17, 21.9))
		return (self.Emax - self.Emin) * En + self.Emin

	def elastance_finite_difference(self, t):
		"""
		Calculates finite-difference approximation of elastance derivative. In class I showed another method
		that calculated the derivative analytically, but I've removed it to keep things simple.

		:param t: time (needed because elastance is a function of time)
		:return: finite-difference approximation of time derivative of time-varying elastance
		"""
		dt = .0001
		forward_time = t + dt
		backward_time = max(0, t - dt)  # small negative times are wrapped to end of cycle
		forward = self.elastance(forward_time)
		backward = self.elastance(backward_time)
		return (forward - backward) / (2 * dt)

	def simulate(self, total_time):
		"""
		:param total_time: time taken to simulate (s)
		:return: time, state (times at which the state is estimated, state vector at each time)
		"""
		# Put all the blood pressure in the atria as an initial condition.
		x0 = [0, self.non_slack_blood_volume / self.C2, 0, 0]
		t_span = (0, total_time)
		dt = 0.001

		sol = solve_ivp(self.get_derivative, t_span, x0, max_step=dt)
		return sol.t, sol.y.T

	def _get_normalized_time(self, t):
		"""
		:param t: time (s)
		:return: time normalized to self.Tmax (duration of ventricular contraction)
		"""
		return (t % self.tc) / self.Tmax

	def left_ventricular_blood_volume(self, t, x):
		"""
		:param t: time (needed because elastance is a function of time)
		:param x: state variables [ventricular pressure; atrial pressure; arterial pressure; aortic flow]
		:return: time-varying blood volume in left ventricle
		"""
		# Assume a slack ventricular volume of 20mL.
		slack_volume = 20
		return x[:, 0] / self.elastance(t) + slack_volume


def plot_pressure_graphs(model, time, states):
	aortic_pressure = states[:, 2] + states[:, 3] * model.R4

	plt.title('States of Circulation versus Time')
	plt.plot(time, states[:, 0], 'k', label='Ventricular Pressure')
	plt.plot(time, states[:, 1], 'g', label='Atrial Pressure')
	plt.plot(time, states[:, 2], 'r', label='Arterial Pressure')
	plt.plot(time, aortic_pressure, 'c', label='Aortic Pressure')
	plt.ylabel('Pressure (mmHg)')
	plt.xlabel('Time (s)')
	plt.legend(loc='upper left')
	plt.show()


def plot_pressure_volume_loops(model):
	initial_R1 = model.R1
	initial_R3 = model.R3

	simulation_time = 10
	dt = 0.001  # Same as in Circulation.simulate()
	start_index = int(model.tc * 5 / dt)  # Start at the index after 5 contractions

	plt.title('Pressure Volume Loops')

	# Normal
	time, states = model.simulate(simulation_time)
	plt.plot(model.left_ventricular_blood_volume(time[start_index:],
			 states[start_index:]), states[start_index:, 0], 'k', label='Normal')

	# High Systemic Resistance
	model.R1 = 2
	time, states = model.simulate(simulation_time)
	plt.plot(model.left_ventricular_blood_volume(time[start_index:], states[start_index:]),
			 states[start_index:, 0], 'g', label='High Systemic Resistance')

	# Aortic Stenosis
	model.R1 = 0.5
	model.R3 = 0.2
	time, states = model.simulate(simulation_time)
	plt.plot(model.left_ventricular_blood_volume(time[start_index:],
			 states[start_index:]), states[start_index:, 0], 'r',
			 label='Aortic Stenosis')

	# Reset Initial Values
	model.R1 = initial_R1
	model.R3 = initial_R3

	plt.ylabel('Pressure (mmHg)')
	plt.xlabel('Volume (ml)')
	plt.legend(loc="upper left")
	plt.show()


if __name__ == '__main__':
	################ Q1 ################
	model = Circulation(75, 2.0, 0.06)

	################ Q2 ################
	t, x = model.simulate(5)
	plot_pressure_graphs(model, t, x)

	################ Q3 ################
	print('Left Ventricular Blood Volume: {}'.format(model.left_ventricular_blood_volume(t, x)))

	################ Q4 ################
	plot_pressure_volume_loops(model)
