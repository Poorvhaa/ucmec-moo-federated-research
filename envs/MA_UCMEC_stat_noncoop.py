import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
import cvxpy as cp
from typing import Optional, Dict, Any

from moo.config import PRESETS, DEFAULT_PHYSICAL_PARAMS, DEFAULT_REFERENCE_SCALES
from moo.metrics import MetricContract, StepMetricSnapshot
from moo.scalarization import MultiObjectiveScalarizer
from moo.diagnostics import SolverDiagnosticsTracker, solve_cpu_allocation


class MA_UCMEC_stat_noncoop(object):
    def __init__(
        self,
        render: bool = False,
        preset_name: str = "legacy",
        moo_weights: Optional[Dict[str, float]] = None,
        reference_scales: Optional[Dict[str, float]] = None,
        physical_params: Optional[Dict[str, Any]] = None,
        use_corrected_cpu_alloc: Optional[bool] = None,
        max_steps: int = 200,
        solver_name: str = "CLARABEL",
        failure_dump_dir: Optional[str] = "experiments_storage/solver_failures",
    ):
        # Initialization
        self.is_mobile = False
        gym.logger.set_level(40)
        np.random.seed(37)
        self.max_steps = max_steps
        self.solver_name = solver_name
        self.preset_name = preset_name
        
        # Configure preset & MOO settings
        preset_cfg = PRESETS.get(preset_name, PRESETS["legacy"])
        selected_mode = preset_cfg.get("mode", "legacy")
        weights = moo_weights if moo_weights is not None else preset_cfg.get("weights", {"w_delay": 1.0, "w_energy": 0.0, "w_overhead": 0.0, "w_reliability": 0.0})
        
        if use_corrected_cpu_alloc is not None:
            self.use_corrected_cpu_alloc = use_corrected_cpu_alloc
        else:
            self.use_corrected_cpu_alloc = preset_cfg.get("use_corrected_cpu_alloc", True)

        self.metric_contract = MetricContract(physical_params)
        self.scalarizer = MultiObjectiveScalarizer(
            weights=weights,
            reference_scales=reference_scales,
            mode=selected_mode
        )
        self.solver_tracker = SolverDiagnosticsTracker(failure_dump_dir=failure_dump_dir)
        self.last_metric_snapshot: Optional[StepMetricSnapshot] = None
        self.last_solver_diagnostics: list = []
        self.M = 60  # number of users
        self.N = 60  # number of APs
        self.varsig = 16  # number of antennas of each AP
        self.K = 3  # number of CPUs
        self.P_max = 0.1  # maximum transmit power of user / pilot power
        self.M_sim = 10  # number of users for simulation
        self.N_sim = 50  # number of APs for simulation

        # locations of users and APs
        self.locations_users = np.random.random_sample([self.M, 2]) * 900  # 2-D location of users
        self.locations_aps = np.random.random_sample([self.N, 2]) * 900  # 2-D location of APs

        # location of 3 CPUs
        self.locations_cpu = np.zeros([3, 2])
        self.locations_cpu[0, 0] = 300
        self.locations_cpu[0, 1] = 300
        self.locations_cpu[1, 0] = 600
        self.locations_cpu[1, 1] = 300
        self.locations_cpu[2, 0] = 450
        self.locations_cpu[2, 1] = 600
        # self.locations_cpu[3, 0] = 600
        # self.locations_cpu[3, 1] = 600

        # calculate distance between APs and users MxN matrix
        self.distance_matrix = np.zeros([self.M, self.N])
        self.distance_matrix_front = np.zeros([self.N, self.K])
        for i in range(self.M):
            for j in range(self.N):
                self.distance_matrix[i, j] = math.sqrt((self.locations_users[i, 0] - self.locations_aps[j, 0]) ** 2
                                                       + (self.locations_users[i, 1] - self.locations_aps[j, 1]) ** 2)

        for i in range(self.N):
            for j in range(self.K):
                self.distance_matrix_front[i, j] = math.sqrt((self.locations_aps[i, 0] - self.locations_cpu[j, 0]) ** 2
                                                             + (self.locations_aps[i, 1] - self.locations_cpu[
                    j, 1]) ** 2)

        # edge computing parameter
        # user parameter
        self.C_user = np.random.uniform(2e8, 5e8, [1, self.M])  # computing resource of users  in Hz
        self.Task_size = np.random.uniform(50000, 100000, [1, self.M])  # task size in bit
        self.Task_density = np.random.uniform(500, 1000, [1, self.M])  # task density cpu cycles per bit
        # Task_max_delay = np.random.uniform(2, 5, [1, M])  # task max delay in second
        self.cluster_size = 1 # AP cluster size

        # edge server parameter
        self.C_edge = np.random.uniform(10e9, 20e9, [self.K, 1])  # computing resource of edge server in CPU

        # access channel
        self.tau_c = 0.1  # coherence time = 100ms
        self.access_chan = np.zeros([self.M, self.N, self.varsig], dtype=complex)  # complex channel
        self.bandwidth_a = 2e6  # bandwidth of access channel
        self.kappa_1 = np.random.rand(1, self.N)  # parameter in Eq. (5)
        self.kappa_2 = np.random.rand(1, self.M)  # parameter in Eq. (5)
        self.f_carrier = 1.9e9  # carrier frequency in Hz
        self.h_ap = 15  # antenna height of AP
        self.h_user = 1.65  # antenna height of user
        # L = 46.3 + 33.9 * np.log10(f_carrier / 1000) - 13.82 * np.log10(h_ap) - (
        #        1.11 * np.log10(f_carrier / 1000) - 0.7) * h_user + 1.56 * np.log10(f_carrier / 1000) - 0.8
        self.L = 140.7
        self.d_0 = 10  # path-loss distance threshold
        self.d_1 = 50  # path-loss distance threshold
        self.PL = np.zeros([self.M, self.N])  # path-loss in dB
        self.beta = np.zeros([self.M, self.N])  # large scale fading
        self.gamma = np.zeros([self.M, self.N])
        self.sigma_s = 8  # standard deviation of shadow fading (dB)
        self.delta = 0.5  # parameter in Eq. (5)
        self.mu = np.zeros([self.M, self.N])  # shadow fading parameter
        self.h = np.zeros([self.M, self.N, self.varsig], dtype=complex)  # small scale fading
        self.noise_access = 3.9810717055349565e-21 * self.bandwidth_a  # noise of access channel -> -174 dbm/Hz

        for i in range(self.M):
            for j in range(self.N):
                # three slope path-loss model
                if self.distance_matrix[i, j] > self.d_1:
                    self.PL[i, j] = -self.L - 35 * np.log10(self.distance_matrix[i, j] / 1000)
                elif self.d_0 <= self.distance_matrix[i, j] <= self.d_1:
                    self.PL[i, j] = -self.L - 10 * np.log10(
                        (self.d_1 / 1000) ** 1.5 * (self.distance_matrix[i, j] / 1000) ** 2)
                else:
                    self.PL[i, j] = -self.L - 10 * np.log10((self.d_1 / 1000) ** 1.5 * (self.d_0 / 1000) ** 2)

                # Eq. (5) shadow fading computation
                self.mu[i, j] = math.sqrt(self.delta) * self.kappa_1[0, j] + math.sqrt(1 - self.delta) * self.kappa_2[
                    0, i]  # MxN matrix as Eq. (5)

                # Eq. (2) channel computation
                self.beta[i, j] = pow(10, self.PL[i, j] / 10) * pow(10, (self.sigma_s * self.mu[i, j]) / 10)
                for k in range(self.varsig):
                    self.h[i, j, k] = np.random.normal(loc=0, scale=0.5) + 1j * np.random.normal(loc=0, scale=0.5)
                    self.access_chan[i, j, k] = np.sqrt(self.beta[i, j]) * self.h[i, j, k]

        # fronthaul channel
        # front_chan = np.zeros([N, K])
        self.bandwidth_f = 2e9  # bandwidth of fronthaul channel
        self.epsilon = 6e-4  # blockage density
        self.p_ap = 1  # transmit power of APs (30 dBm = 1 W)
        self.alpha_los = 2.5  # path-loss exponent for LOS links
        self.alpha_nlos = 4  # path-loss exponent for NLOS links
        self.psi_los = 3  # Nakagami fading parameter for LOS links
        self.psi_nlos = 2  # Nakagami fading parameter for NLOS links
        self.noise_front = 1.380649 * 10e-23 * 290 * 9 * self.bandwidth_f  # fronthaul channel noise variance
        self.G = np.zeros([self.N, self.K])  # random antenna gain
        self.fai = math.pi / 6  # Main lobe beamwidth
        self.Gm = 63.1  # Directivity gain of main lobes
        self.Gs = 0.631  # Directivity gain of side lobes
        self.Gain = np.array(
            [self.Gs * self.Gs, self.Gm * self.Gm, self.Gm * self.Gs])  # random antenna gain in Eq. (7)
        self.Gain_pro = np.array(
            [(self.fai / (2 * math.pi)) ** 2, 2 * self.fai * (2 * math.pi - self.fai) / (2 * math.pi) ** 2,
             ((2 * math.pi - self.fai) / (2 * math.pi)) ** 2])

        self.P_los = np.zeros([self.N, self.K])  # probability of LOS links
        self.link_type = np.zeros([self.N, self.K])  # type of fronthaul links
        for i in range(self.N):
            for j in range(self.K):
                self.P_los[i, j] = np.exp(-self.epsilon * self.distance_matrix_front[i, j] / 1000)
                self.link_type[i, j] = np.random.choice([0, 1], p=[self.P_los[i, j],
                                                                   1 - self.P_los[i, j]])  # 0 for LOS, 1 for NLOS
                # if link_type[i, j] == 0:  # LOS link
                #     front_chan[i, j] = np.random.gamma(2, 1 / psi_los)  # Nakagami channel gain
                # else:  # NLOS link
                #     front_chan[i, j] = np.random.gamma(2, 1 / psi_nlos)  # Nakagami channel gain
                self.G[i, j] = np.random.choice(self.Gain, p=self.Gain_pro.ravel())
        # pilot assignment
        self.tau_p = self.M  # length of pilot symbol
        self.pilot_matrix = np.zeros([self.M, self.tau_p])
        for i in range(self.M):
            self.pilot_index = i
            self.pilot_matrix[i, self.pilot_index] = 1

        # MMSE channel estimation
        # receive_pilot = np.zeros([self.N, self.varsig, self.tau_p], dtype=complex)
        # access_chan_estimate = np.zeros([M, N, varsig], dtype=complex)
        self.theta = np.zeros([self.M, self.N])

        for i in range(self.M):
            for j in range(self.N):
                self.theta[i, j] = self.tau_p * self.P_max * (self.beta[i, j] ** 2) / (
                        self.tau_p * self.P_max * self.beta[i, j] + self.noise_access)

        # parameter init
        self.n_agents = self.M_sim
        self.agent_num = self.n_agents
        self.obs_dim = 3  # set the observation dimension of agents
        self.action_dim = 10
        self._render = render
        # action space: [omega_1,omega_2,...,omega_K,p]  K+1 continuous vector for each agent
        # a in {0,1,2,3,4}, p in {0, 1, 2, 3, 4} (totally 5 levels (p+1)/5*100 mW)
        self.omega_last = np.zeros([self.M_sim])
        self.p_last = np.zeros([self.M_sim])
        self.delay_last = np.zeros([self.M_sim, 1])
        self.action_space = spaces.Tuple(tuple([spaces.Discrete(10)] * self.n_agents))
        # state space: [r_1(t-1),r_2(t-1),...,r_M(t-1)]  1xM continuous vector. -> uplink rate
        # r in [0, 10e8]
        self.obs_low = np.array([0, 0, 0])
        self.obs_high = np.array([2, self.P_max, 0.2])
        # self.observation_space = spaces.MultiDiscrete([500] * self.M_sim)
        # self.observation_space = spaces.Box(low=self.r_low, high=self.r_high, shape=(self.M_sim,), dtype=np.float32)
        self.observation_space = spaces.Tuple(tuple(
            [spaces.Box(low=self.obs_low, high=self.obs_high, shape=(self.obs_dim,),
                        dtype=np.float32)] * self.n_agents))
        # self.np_random = None
        self.uplink_rate_access_b = np.zeros([self.M_sim, 1])
        self.step_num = 0
        self.cluster_matrix = self.cluster()

    def action_mapping(self, action_agent):
        omega_agent = 0
        p_agent = 0
        # Transform the action space form MultiDiscrete to Discrete (1+3*3=10 cases)
        if action_agent[0] == 1:  # local processing
            omega_agent = 0
            p_agent = 0
        elif action_agent[1] == 1:
            omega_agent = 1
            p_agent = 1
        elif action_agent[2] == 1:
            omega_agent = 1
            p_agent = 2
        elif action_agent[3] == 1:
            omega_agent = 1
            p_agent = 3
        elif action_agent[4] == 1:
            omega_agent = 2
            p_agent = 1
        elif action_agent[5] == 1:
            omega_agent = 2
            p_agent = 2
        elif action_agent[6] == 1:
            omega_agent = 2
            p_agent = 3
        elif action_agent[7] == 1:
            omega_agent = 3
            p_agent = 1
        elif action_agent[8] == 1:
            omega_agent = 3
            p_agent = 2
        elif action_agent[9] == 1:
            omega_agent = 3
            p_agent = 3
        return omega_agent, p_agent

    def cluster(self):
        # AP cluster
        cluster_matrix = np.zeros([self.M_sim, self.N_sim])  # AP serve user when value = 1, otherwise 0
        max_h_index_list = np.zeros(
            [self.M_sim, self.N_sim])  # sort access channel (large-scale fading) from high to low
        ap_index_list = np.zeros([self.M_sim, self.cluster_size])  # obtain the index of APs serve for each user
        for i in range(self.M_sim):
            for j in range(self.N_sim):
                max_h_index_list[i, :] = self.beta[i, 0:self.N_sim].argsort()
                max_h_index_list[i, :] = max_h_index_list[i, :][::-1]
                ap_index_list[i, :] = max_h_index_list[i, 0:self.cluster_size]
                for k in range(self.cluster_size):
                    cluster_matrix[i, int(ap_index_list[i, k])] = 1
        return cluster_matrix

    def uplink_rate_cal(self, p, omega):  # calculate the uplink transmit rate in Eq. (12)
        SINR_access = np.zeros([self.M_sim, 1])
        uplink_rate_access = np.zeros([self.M_sim, 1])
        SINR_access_mole = np.zeros([self.M_sim, 1])
        SINR_access_inter = np.zeros([self.M_sim, 1])
        SINR_access_noise = np.zeros([self.M_sim, 1])

        for i in range(self.M_sim):
            if omega[i] == 0:  # Local processing
                continue
            else:
                SINR_access_inter[i, 0] = 0  # Interferencecontinue
                SINR_access_mole[i, 0] = 0  # Useful symbol
                SINR_access_noise[i, 0] = 0  # Noise
                for j in range(self.N_sim):
                    if self.cluster_matrix[i, j] == 1:
                        SINR_access_mole[i, 0] = SINR_access_mole[i, 0] + self.theta[i, j]
                        SINR_access_noise[i, 0] = self.noise_access * self.theta[i, j]
                SINR_access_mole[i, 0] = (SINR_access_mole[i, 0] ** 2) * p[i] * self.varsig

                for k in range(self.M_sim):
                    if k == i or omega[k] == 0:
                        continue
                    else:
                        for j in range(self.N_sim):
                            if self.cluster_matrix[i, j] == 1:
                                SINR_access_inter[i, 0] = SINR_access_inter[i, 0] + self.theta[i, j] * self.beta[k, j] * \
                                                          p[k]
                SINR_access[i, 0] = SINR_access_mole[i, 0] / (SINR_access_inter[i, 0] + SINR_access_noise[i, 0])
                uplink_rate_access[i, 0] = self.bandwidth_a * np.log2(1 + SINR_access[i, 0])
        return uplink_rate_access

    def front_rate_cal(self, omega):
        chi = np.zeros([self.N_sim, self.K])  # whether an AP transmit symbol to a CPU or not
        SINR_front = np.zeros([self.N_sim, self.K])  # SINR in Eq. (7)
        front_rate = np.zeros([self.N_sim, self.K])  # Eq. (12)
        front_rate_user = np.zeros([self.M_sim, self.N_sim])
        I_sum = 0  # total sum of fronthaul interference
        for i in range(self.M_sim):
            if omega[i] == 0:
                continue
            CPU_id = int(omega[i] - 1)
            for j in range(self.N_sim):
                if self.cluster_matrix[i, j] == 1:  # This AP is belonged to the cluster of user i
                    chi[j, CPU_id] = 1

        for i in range(self.N_sim):
            for j in range(self.K):
                if chi[i, j] == 1:
                    if self.link_type[j, j] == 0:  # LOS link
                        I_sum = I_sum + self.p_ap * pow(self.distance_matrix_front[i, j] / 1000, -self.alpha_los)
                    else:
                        I_sum = I_sum + self.p_ap * pow(self.distance_matrix_front[i, j] / 1000, -self.alpha_nlos)
                else:
                    pass

        for i in range(self.N_sim):
            for j in range(self.K):
                if chi[i, j] == 1:
                    if self.link_type[i, j] == 0:  # LOS link
                        SINR_front_mole = self.p_ap * self.G[i, j] * pow(self.distance_matrix_front[i, j] / 1000,
                                                                         -self.alpha_los)
                    else:
                        SINR_front_mole = self.p_ap * self.G[i, j] * pow(self.distance_matrix_front[i, j] / 1000,
                                                                         self.alpha_nlos)
                    SINR_front[i, j] = SINR_front_mole / (I_sum - SINR_front_mole / self.G[i, j] + self.noise_front)
                    front_rate[i, j] = self.bandwidth_f * np.log2(1 + SINR_front[i, j])

        for i in range(self.M_sim):
            if omega[i] == 0:
                pass
            else:
                CPU_id = int(omega[i] - 1)
                for j in range(self.N_sim):
                    if self.cluster_matrix[i, j] == 1:
                        front_rate_user[i, j] = front_rate[j, CPU_id]

        return front_rate_user

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def reset(self):
        self.step_num = 0
        self.omega_last.fill(0)
        self.p_last.fill(0)
        self.delay_last.fill(0)
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.uniform(low=self.obs_low, high=self.obs_high, size=(self.obs_dim,))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, action):
        self.step_num += 1

        if self.is_mobile:
            max_speed = 15 * self.tau_c  # maximum speed of users (m / tau_c second)
            min_speed = 5 * self.tau_c  # maximum speed of users (m / tau_c second)
            destination_users = np.random.random_sample([self.M, 2]) * 900
            user_speed = np.random.uniform(min_speed, max_speed, [self.M, 1])
            for i in range(self.M):
                slope = np.sqrt(np.abs(self.locations_users[i, 0] - destination_users[i, 0]) ** 2 + np.abs(
                    self.locations_users[i, 1] - destination_users[i, 1]) ** 2)
                self.locations_users[i, 0] = self.locations_users[i, 0] + user_speed[i, 0] * np.abs(
                    destination_users[i, 0] - self.locations_users[i, 0]) / slope
                self.locations_users[i, 1] = self.locations_users[i, 1] + user_speed[i, 0] * np.abs(
                    destination_users[i, 1] - self.locations_users[i, 1]) / slope

            for i in range(self.M):
                for j in range(self.N):
                    self.distance_matrix[i, j] = math.sqrt((self.locations_users[i, 0] - self.locations_aps[j, 0]) ** 2
                                                           + (self.locations_users[i, 1] - self.locations_aps[
                        j, 1]) ** 2)
            # distance
            for i in range(self.M):
                for j in range(self.N):
                    # three slope path-loss model
                    if self.distance_matrix[i, j] > self.d_1:
                        self.PL[i, j] = -self.L - 35 * np.log10(self.distance_matrix[i, j] / 1000)
                    elif self.d_0 <= self.distance_matrix[i, j] <= self.d_1:
                        self.PL[i, j] = -self.L - 10 * np.log10(
                            (self.d_1 / 1000) ** 1.5 * (self.distance_matrix[i, j] / 1000) ** 2)
                    else:
                        self.PL[i, j] = -self.L - 10 * np.log10((self.d_1 / 1000) ** 1.5 * (self.d_0 / 1000) ** 2)

            # access channel
            kappa_1 = np.random.rand(1, self.N)  # parameter in Eq. (5)
            kappa_2 = np.random.rand(1, self.M)  # parameter in Eq. (5)

            for i in range(self.M):
                for j in range(self.N):
                    # Eq. (5) shadow fading computation
                    self.mu[i, j] = math.sqrt(self.delta) * kappa_1[0, j] + math.sqrt(1 - self.delta) * kappa_2[
                        0, i]  # MxN matrix as Eq. (5)

                    # Eq. (2) channel computation
                    self.beta[i, j] = pow(10, self.PL[i, j] / 10) * pow(10, (self.sigma_s * self.mu[i, j]) / 10)
                    for k in range(self.varsig):
                        self.h[i, j, k] = np.random.normal(loc=0, scale=0.5) + 1j * np.random.normal(loc=0, scale=0.5)
                        self.access_chan[i, j, k] = np.sqrt(self.beta[i, j]) * self.h[i, j, k]

            # MMSE channel estimation
            theta = np.zeros([self.M, self.N])
            for i in range(self.M):
                for j in range(self.N):
                    theta[i, j] = self.tau_p * self.P_max * (self.beta[i, j] ** 2) / (
                            self.tau_p * self.P_max * self.beta[i, j] + self.noise_access)
        # obtain and clip the action
        omega_current = np.zeros([self.M_sim])
        p_current = np.zeros([self.M_sim])
        p_level = self.P_max / 4
        # print("Step Index:", self.step_num)
        for i in range(self.M_sim):
            omega_current[i], p_current[i] = self.action_mapping(action[i])
            p_current[i] = (p_current[i] + 1) * p_level
            # print("Chosen CPU ID:", omega_current)
            # print("Power:", p_current)

        uplink_rate_access = self.uplink_rate_cal(p_current, omega_current)
        front_rate_user = self.front_rate_cal(omega_current)
        self.uplink_rate_access_b = uplink_rate_access
        # print("Fronthaul Rate", front_rate_user)
        # print("Uplink Rate (Mbps):", uplink_rate_access / 10e6)
        # print("Average Uplink Rate (Mbps):", np.sum(uplink_rate_access) / (np.count_nonzero(omega_current) * 10e6))

        # local computing delay
        local_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] == 0:
                local_delay[i, 0] = self.Task_density[0, i] * self.Task_size[0, i] / self.C_user[0, i]

        # uplink delay
        uplink_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] != 0:
                uplink_delay_list = np.zeros([self.cluster_size, 1])
                for j in range(self.cluster_size):
                    uplink_delay_list[j, 0] = self.Task_size[0, i] / uplink_rate_access[i, 0]
                uplink_delay[i, 0] = np.max(uplink_delay_list)

        # fronthaul delay
        front_delay_matrix = np.zeros([self.M_sim, self.cluster_size])
        front_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] != 0:
                for j in range(self.N_sim):
                    for k in range(self.cluster_size):
                        if self.cluster_matrix[i, j] == 1:
                            front_delay_matrix[i, k] = self.Task_size[0, i] / front_rate_user[i, j]
                front_delay[i, 0] = np.max(front_delay_matrix[i, :])

        # processing delay calculation
        # solve convex problem according to Eq. (24)
        task_mat = np.zeros([self.M_sim, self.K])
        for i in range(self.M_sim):
            if omega_current[i] != 0:
                CPU_id = int(omega_current[i] - 1)
                task_mat[i, CPU_id] = self.Task_size[0, i] * self.Task_density[0, i]

        # Each CPU solves a resource allocation optimization problem
        actual_C = np.zeros([self.M_sim, self.K])
        step_solver_diagnostics = []
        for i in range(self.K):
            serve_user_id = []
            serve_user_task = []
            _local_delay = []
            _front_delay = []
            _uplink_delay = []

            for j in range(self.M_sim):
                if task_mat[j, i] != 0:
                    serve_user_id.append(j)
                    serve_user_task.append(task_mat[j, i])
                    _local_delay.append(local_delay[j, 0])
                    _uplink_delay.append(uplink_delay[j, 0])
                    _front_delay.append(front_delay[j, 0])
            if len(serve_user_id) == 0:
                continue
            capacity = float(self.C_edge[i, 0])
            
            allocated_cpu, diag = solve_cpu_allocation(
                cpu_id=i,
                capacity=capacity,
                user_ids=serve_user_id,
                tasks=np.asarray(serve_user_task, dtype=float),
                local_delays=np.asarray(_local_delay, dtype=float),
                uplink_delays=np.asarray(_uplink_delay, dtype=float),
                front_delays=np.asarray(_front_delay, dtype=float),
                use_corrected_cpu_alloc=self.use_corrected_cpu_alloc,
                solver_name=self.solver_name,
                tracker=self.solver_tracker,
            )
            step_solver_diagnostics.append(diag.to_dict())
            for k, user_id in enumerate(serve_user_id):
                actual_C[user_id, i] = allocated_cpu[k]

        self.last_solver_diagnostics = step_solver_diagnostics

        actual_process_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] != 0:
                CPU_id = int(omega_current[i] - 1)
                actual_process_delay[i, 0] = task_mat[i, CPU_id] / max(1e-6, float(np.sum(actual_C[i, :])))

        # Total delay calculation: mutually exclusive paths
        total_delay = np.zeros([self.M_sim, 1])
        for i in range(self.M_sim):
            if omega_current[i] == 0:
                total_delay[i, 0] = local_delay[i, 0]
            else:
                total_delay[i, 0] = front_delay[i, 0] + uplink_delay[i, 0] + actual_process_delay[i, 0]

        # Multi-objective metrics evaluation
        metric_snapshot = self.metric_contract.compute_step_metrics(
            omega=omega_current,
            p=p_current,
            local_delay=local_delay,
            uplink_delay=uplink_delay,
            front_delay=front_delay,
            process_delay=actual_process_delay,
            task_size_bits=self.Task_size,
            task_density=self.Task_density,
            c_user=self.C_user,
            c_actual=actual_C,
            sinr_access=SINR_access if 'SINR_access' in locals() else None,
        )
        self.last_metric_snapshot = metric_snapshot

        # Per-user arrays for scalarization
        per_user_delay_s = total_delay.flatten()
        per_user_energy_j = np.zeros(self.M_sim, dtype=float)
        kappa_u = float(self.metric_contract.params["kappa_user"])
        for i in range(self.M_sim):
            if omega_current[i] == 0:
                cycles = float(self.Task_density[0, i] * self.Task_size[0, i])
                per_user_energy_j[i] = kappa_u * cycles * float(self.C_user[0, i] ** 2)
            else:
                per_user_energy_j[i] = float(p_current[i] * uplink_delay[i, 0])

        per_user_overhead_b = np.zeros(self.M_sim, dtype=float)
        fixed_sig = (
            float(self.metric_contract.params["csi_pilot_bytes"])
            + float(self.metric_contract.params["offload_request_bytes"])
            + float(self.metric_contract.params["scheduling_grant_bytes"])
            + float(self.metric_contract.params["fronthaul_signaling_bytes"])
            + float(self.metric_contract.params["ack_bytes"])
        )
        for i in range(self.M_sim):
            if omega_current[i] != 0:
                per_user_overhead_b[i] = fixed_sig

        deadline_s = float(self.metric_contract.params["task_deadline_s"])
        per_user_success = per_user_delay_s <= deadline_s

        reward, reward_diagnostics = self.scalarizer.compute_agent_rewards(
            per_user_delay_s=per_user_delay_s,
            per_user_energy_j=per_user_energy_j,
            per_user_overhead_b=per_user_overhead_b,
            per_user_success=per_user_success,
            tau_c=self.tau_c,
        )

        is_truncated = self.step_num >= self.max_steps
        done = [1 if is_truncated else 0] * self.M_sim

        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []

        for i in range(self.agent_num):
            observation = np.zeros(self.obs_dim, dtype=np.float32)
            if self.step_num == 1:
                observation[0] = 0.0
                observation[1] = 0.0
                observation[2] = 0.0
            else:
                observation[0] = float(self.omega_last[i])
                observation[1] = float(self.p_last[i])
                observation[2] = float(self.delay_last[i, 0])

            info = {
                "individual_reward": float(reward[i, 0]),
                "bad_transition": is_truncated,
                "TimeLimit.truncated": is_truncated,
                "delay_s": float(per_user_delay_s[i]),
                "delay_ms": float(per_user_delay_s[i] * 1000.0),
                "energy_j": float(per_user_energy_j[i]),
                "energy_mj": float(per_user_energy_j[i] * 1000.0),
                "overhead_bytes": float(per_user_overhead_b[i]),
                "task_success": bool(per_user_success[i]),
                "step_metrics": metric_snapshot.to_dict(),
                "reward_diagnostics": reward_diagnostics,
                "solver_diagnostics": step_solver_diagnostics,
            }

            sub_agent_obs.append(observation)
            sub_agent_reward.append(reward[i])
            sub_agent_done.append(done[i])
            sub_agent_info.append(info)

        self.delay_last = total_delay
        self.omega_last = omega_current
        self.p_last = p_current
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]


# if __name__ == "__main__":
#     env = MA_UCMEC_Static(render=False)
#     # check_env(env)
#     obs = env.reset()
#     n_steps = 50
#     for _ in range(n_steps):
#         # Random action
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         if np.all(done):
#             obs = env.reset()
#         # print(f"state: {obs} \n")
#         print(f"action : {action}, reward : {reward}")
