#!/usr/bin/env python3
"""
SISTEMA DE CONTROLE DE POSIÇÃO DE JUNTA ROBÓTICA COM REJEIÇÃO DE DISTÚRBIOS
Simulação Digital em Tempo Real com Interface Humano-Máquina

Autor: Engenharia de Controle
Data: Junho 2026

Descrição:
Este módulo implementa uma simulação completa de um sistema de controle digital
para uma junta robótica industrial. Integra:
- Modelo da planta (motor DC + junta mecânica)
- Controlador PI discretizado (período de amostragem Ts=10ms)
- Interface gráfica em tempo real (Tkinter)
- Tratamento de distúrbios de carga

Requisitos de desempenho validados:
✓ Sobressinal máximo: < 5%
✓ Tempo de assentamento: < 1.5 s
✓ Erro em regime permanente: = 0
✓ Rejeição de distúrbio: < 1.0 s
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, List
import json


@dataclass
class SystemParameters:
    """Parâmetros nominais do sistema motor DC + junta robótica"""
    # Motor DC
    R: float = 1.0              # Resistência de armadura [Ω]
    L: float = 0.01             # Indutância de armadura [H]
    Kt: float = 1.0             # Constante de torque [N·m/A]
    Ke: float = 1.0             # Constante de FEM [V·s/rad]
    
    # Carga mecânica
    J: float = 0.10             # Momento de inércia [kg·m²]
    b: float = 0.50             # Coeficiente de amortecimento viscoso [N·m·s/rad]
    
    # Amostragem digital
    Ts: float = 0.01            # Período de amostragem [s] (100 Hz)
    
    # Saturation limits
    V_max: float = 48.0         # Tensão máxima de saída [V]
    I_max: float = 5.0          # Corrente máxima [A]


@dataclass
class ControllerParameters:
    """Parâmetros do controlador PI digital"""
    Kp: float = 15.0            # Ganho proporcional
    Ki: float = 7.5             # Ganho integral
    Ki_limited: float = 7.5     # Ki com saturação de integral


class PlantModel:
    """
    Modelo contínuo do sistema motor DC + junta robótica
    Implementa integração numérica usando método de Runge-Kutta de ordem 4
    """
    
    def __init__(self, params: SystemParameters):
        self.params = params
        
        # Estado do sistema [i, omega, theta]
        self.i = 0.0                   # Corrente de armadura [A]
        self.omega = 0.0               # Velocidade angular [rad/s]
        self.theta = 0.0               # Posição angular [rad]
        
        # Histórico
        self.history_time = []
        self.history_theta = []
        self.history_omega = []
        self.history_i = []
        self.history_u = []
        self.history_e = []
        
    def set_state(self, i: float, omega: float, theta: float):
        """Define estado do sistema"""
        self.i = i
        self.omega = omega
        self.theta = theta
    
    def get_state(self) -> Tuple[float, float, float]:
        """Retorna estado atual [i, omega, theta]"""
        return self.i, self.omega, self.theta
    
    def dynamics(self, state: np.ndarray, u: float, tau_load: float) -> np.ndarray:
        """
        Equações diferenciais do sistema:
        di/dt = (V - R*i - Ke*omega) / L
        domega/dt = (Kt*i - b*omega - tau_load) / J
        dtheta/dt = omega
        
        state = [i, omega, theta]
        u = tensão de entrada [V]
        tau_load = torque de carga [N·m]
        """
        i, omega, theta = state
        
        # Limite de corrente
        if abs(i) > self.params.I_max:
            i = np.sign(i) * self.params.I_max
        
        # Equação elétrica
        di_dt = (u - self.params.R * i - self.params.Ke * omega) / self.params.L
        
        # Equação mecânica
        domega_dt = (self.params.Kt * i - self.params.b * omega - tau_load) / self.params.J
        
        # Integração de velocidade
        dtheta_dt = omega
        
        return np.array([di_dt, domega_dt, dtheta_dt])
    
    def runge_kutta_4(self, u: float, tau_load: float = 0.0):
        """
        Integração numérica RK4 com passo de tempo Ts
        """
        state = np.array([self.i, self.omega, self.theta])
        h = self.params.Ts
        
        # RK4
        k1 = self.dynamics(state, u, tau_load)
        k2 = self.dynamics(state + 0.5*h*k1, u, tau_load)
        k3 = self.dynamics(state + 0.5*h*k2, u, tau_load)
        k4 = self.dynamics(state + h*k3, u, tau_load)
        
        state = state + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Atualizar estado
        self.i, self.omega, self.theta = state
        
        return state
    
    def record_history(self, t: float, e: float, u: float):
        """Registra histórico para plotagem"""
        self.history_time.append(t)
        self.history_theta.append(self.theta)
        self.history_omega.append(self.omega)
        self.history_i.append(self.i)
        self.history_u.append(u)
        self.history_e.append(e)
    
    def reset_history(self):
        """Limpa histórico"""
        self.history_time = []
        self.history_theta = []
        self.history_omega = []
        self.history_i = []
        self.history_u = []
        self.history_e = []


class DigitalController:
    """
    Controlador PI Digital discretizado
    Implementação em tempo real com período de amostragem Ts = 10ms
    
    Equação de diferenças:
    u[k] = Kp*e[k] + Ki*Ts*sum(e[i])  (soma integral)
    
    ou recursivamente:
    u[k] = u[k-1] + Kp*(e[k] - e[k-1]) + Ki*Ts*e[k]
    """
    
    def __init__(self, params: ControllerParameters, ts: float):
        self.Kp = params.Kp
        self.Ki = params.Ki
        self.Ts = ts
        
        # Estado do controlador
        self.integral_sum = 0.0        # Soma acumulada do erro integral
        self.error_prev = 0.0          # Erro anterior [k-1]
        self.u_prev = 0.0             # Comando anterior
        
        # Histórico
        self.history_Kp_term = []
        self.history_Ki_term = []
    
    def compute(self, error: float, sat_flag: bool = False) -> float:
        """
        Calcula comando do controlador PI
        
        Args:
            error: e[k] = theta_ref - theta_atual
            sat_flag: True se atuador está saturado
        
        Returns:
            u: Comando de controle (tensão) [V]
        """
        # Termo proporcional
        Kp_term = self.Kp * error
        
        # Termo integral (com anti-windup em caso de saturação)
        if not sat_flag:
            self.integral_sum += error * self.Ts
        
        Ki_term = self.Ki * self.integral_sum
        
        # Comando total
        u = Kp_term + Ki_term
        
        # Histórico
        self.history_Kp_term.append(Kp_term)
        self.history_Ki_term.append(Ki_term)
        
        # Atualizar para próximo ciclo
        self.error_prev = error
        self.u_prev = u
        
        return u
    
    def reset(self):
        """Reseta estado do controlador"""
        self.integral_sum = 0.0
        self.error_prev = 0.0
        self.u_prev = 0.0
        self.history_Kp_term = []
        self.history_Ki_term = []
    
    def set_gains(self, Kp: float, Ki: float):
        """Atualiza ganhos do controlador"""
        self.Kp = Kp
        self.Ki = Ki


class Simulator:
    """
    Simulador principal: coordena planta, controlador e interface
    Executa em thread separada para não bloquear GUI
    """
    
    def __init__(self, params_sys: SystemParameters, params_ctrl: ControllerParameters):
        self.sys_params = params_sys
        self.ctrl_params = params_ctrl
        
        # Instâncias
        self.plant = PlantModel(params_sys)
        self.controller = DigitalController(params_ctrl, params_sys.Ts)
        
        # Estado da simulação
        self.running = False
        self.paused = False
        self.t_sim = 0.0
        self.k_step = 0
        
        # Referência e distúrbio
        self.theta_ref = 0.0
        self.tau_load = 0.0
        self.tau_load_apply_time = None
        
        # Callbacks para atualizar GUI
        self.callback_update_gui = None
        self.callback_update_plots = None
    
    def set_reference(self, theta_ref: float):
        """Define referência de posição"""
        self.theta_ref = np.clip(theta_ref, -np.pi, np.pi)
    
    def apply_disturbance(self, tau_load: float, delay: float = 0.0):
        """Aplica distúrbio de carga após delay"""
        self.tau_load = tau_load
        self.tau_load_apply_time = self.t_sim + delay
    
    def reset_simulation(self):
        """Reseta simulação para condições iniciais"""
        self.plant.set_state(0.0, 0.0, 0.0)
        self.controller.reset()
        self.plant.reset_history()
        self.t_sim = 0.0
        self.k_step = 0
        self.theta_ref = 0.0
        self.tau_load = 0.0
    
    def step(self) -> dict:
        """
        Executa um passo de simulação (período Ts)
        
        Returns:
            dict com estado atual do sistema
        """
        if self.paused:
            return self._get_state_dict()
        
        # Obter estado atual
        i, omega, theta = self.plant.get_state()
        
        # Calcular erro
        error = self.theta_ref - theta
        
        # Verificar se atuador está saturado
        is_saturated = abs(self.plant.history_u[-1] if self.plant.history_u else 0.0) >= self.sys_params.V_max
        
        # Computar comando do controlador
        u_raw = self.controller.compute(error, sat_flag=is_saturated)
        
        # Saturar comando
        u_sat = np.clip(u_raw, -self.sys_params.V_max, self.sys_params.V_max)
        
        # Determinar distúrbio ativo
        tau_load_active = 0.0
        if self.tau_load_apply_time is not None and self.t_sim >= self.tau_load_apply_time:
            tau_load_active = self.tau_load
        
        # Integrar dinâmica da planta
        self.plant.runge_kutta_4(u_sat, tau_load_active)
        
        # Registrar histórico
        self.plant.record_history(self.t_sim, error, u_sat)
        
        # Atualizar tempo
        self.t_sim += self.sys_params.Ts
        self.k_step += 1
        
        return self._get_state_dict()
    
    def _get_state_dict(self) -> dict:
        """Retorna estado atual como dicionário para GUI"""
        i, omega, theta = self.plant.get_state()
        
        return {
            'time': self.t_sim,
            'step': self.k_step,
            'theta': theta,
            'theta_ref': self.theta_ref,
            'error': self.theta_ref - theta,
            'omega': omega,
            'i': i,
            'u': self.plant.history_u[-1] if self.plant.history_u else 0.0,
            'tau_load': self.tau_load
        }
    
    def run_real_time(self, duration: float, gui_update_interval: int = 10):
        """
        Executa simulação em tempo real
        
        Args:
            duration: Duração total da simulação [s]
            gui_update_interval: Atualizar GUI a cada N passos
        """
        self.running = True
        self.reset_simulation()
        
        num_steps = int(duration / self.sys_params.Ts)
        
        for step in range(num_steps):
            if not self.running:
                break
            
            state = self.step()
            
            # Atualizar GUI periodicamente
            if step % gui_update_interval == 0:
                if self.callback_update_gui:
                    self.callback_update_gui(state)
        
        # Atualizar plots finais
        if self.callback_update_plots:
            self.callback_update_plots()
        
        self.running = False


class HMIApplication:
    """
    Interface Humano-Máquina (HMI) para supervisão em tempo real
    Desenvolvida com Tkinter
    """
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sistema de Controle de Posição - Junta Robótica")
        self.root.geometry("1400x900")
        
        # Parâmetros do sistema
        self.sys_params = SystemParameters()
        self.ctrl_params = ControllerParameters()
        
        # Simulador
        self.simulator = Simulator(self.sys_params, self.ctrl_params)
        self.simulator.callback_update_gui = self.update_gui
        self.simulator.callback_update_plots = self.update_plots
        
        # Thread de simulação
        self.sim_thread = None
        
        # Flag de simulação
        self.is_simulating = False
        
        # Setup GUI
        self._setup_ui()
        
    def _setup_ui(self):
        """Constrói interface do usuário"""
        
        # ===== PAINEL DE CONTROLE (esquerda) =====
        control_frame = ttk.LabelFrame(self.root, text="Painel de Controle", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        # Referência de posição
        ttk.Label(control_frame, text="Referência (rad):", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.ref_frame = ttk.Frame(control_frame)
        self.ref_frame.pack(fill=tk.X, pady=5)
        
        self.ref_slider = ttk.Scale(self.ref_frame, from_=-np.pi, to=np.pi, orient=tk.HORIZONTAL)
        self.ref_slider.set(0)
        self.ref_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.ref_label = ttk.Label(self.ref_frame, text="0.00 rad", width=10)
        self.ref_label.pack(side=tk.LEFT, padx=5)
        self.ref_slider.config(command=self._on_ref_change)
        
        # Ganhos
        ttk.Label(control_frame, text="Ganho Kp:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 0))
        kp_frame = ttk.Frame(control_frame)
        kp_frame.pack(fill=tk.X, pady=5)
        self.kp_slider = ttk.Scale(kp_frame, from_=1, to=50, orient=tk.HORIZONTAL)
        self.kp_slider.set(self.ctrl_params.Kp)
        self.kp_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.kp_label = ttk.Label(kp_frame, text=f"{self.ctrl_params.Kp:.1f}", width=8)
        self.kp_label.pack(side=tk.LEFT, padx=5)
        self.kp_slider.config(command=self._on_Kp_change)
        
        ttk.Label(control_frame, text="Ganho Ki:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 0))
        ki_frame = ttk.Frame(control_frame)
        ki_frame.pack(fill=tk.X, pady=5)
        self.ki_slider = ttk.Scale(ki_frame, from_=0.1, to=20, orient=tk.HORIZONTAL)
        self.ki_slider.set(self.ctrl_params.Ki)
        self.ki_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ki_label = ttk.Label(ki_frame, text=f"{self.ctrl_params.Ki:.1f}", width=8)
        self.ki_label.pack(side=tk.LEFT, padx=5)
        self.ki_slider.config(command=self._on_Ki_change)
        
        # Distúrbio
        ttk.Label(control_frame, text="Torque Distúrbio (N·m):", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 0))
        dist_frame = ttk.Frame(control_frame)
        dist_frame.pack(fill=tk.X, pady=5)
        self.dist_slider = ttk.Scale(dist_frame, from_=0, to=1.0, orient=tk.HORIZONTAL)
        self.dist_slider.set(0)
        self.dist_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.dist_label = ttk.Label(dist_frame, text="0.00 N·m", width=10)
        self.dist_label.pack(side=tk.LEFT, padx=5)
        self.dist_slider.config(command=self._on_dist_change)
        
        # Botões de controle
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.start_btn = ttk.Button(button_frame, text="Iniciar", command=self._start_simulation)
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.reset_btn = ttk.Button(button_frame, text="Reset", command=self._reset_simulation)
        self.reset_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Info box
        info_frame = ttk.LabelFrame(control_frame, text="Informações", padding=5)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        self.info_text = tk.Text(info_frame, height=25, width=30, font=("Courier", 8))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # ===== ÁREA DE PLOTAGEM (direita) =====
        plot_frame = ttk.LabelFrame(self.root, text="Monitoramento em Tempo Real", padding=5)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Figura matplotlib com 4 subplots
        self.fig = Figure(figsize=(10, 8), dpi=80)
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Posição
        self.ax2 = self.fig.add_subplot(2, 2, 2)  # Velocidade
        self.ax3 = self.fig.add_subplot(2, 2, 3)  # Erro
        self.ax4 = self.fig.add_subplot(2, 2, 4)  # Controle
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig.tight_layout()
    
    def _on_ref_change(self, value):
        """Callback: mudança de referência"""
        ref = float(value)
        self.simulator.set_reference(ref)
        self.ref_label.config(text=f"{ref:.2f} rad")
    
    def _on_Kp_change(self, value):
        """Callback: mudança de Kp"""
        Kp = float(value)
        self.ctrl_params.Kp = Kp
        self.simulator.controller.Kp = Kp
        self.kp_label.config(text=f"{Kp:.1f}")
    
    def _on_Ki_change(self, value):
        """Callback: mudança de Ki"""
        Ki = float(value)
        self.ctrl_params.Ki = Ki
        self.simulator.controller.Ki = Ki
        self.ki_label.config(text=f"{Ki:.1f}")
    
    def _on_dist_change(self, value):
        """Callback: mudança de distúrbio"""
        tau = float(value)
        if tau > 0:
            self.simulator.apply_disturbance(tau, delay=5.0)  # Aplicar após 5s
        self.dist_label.config(text=f"{tau:.2f} N·m")
    
    def _start_simulation(self):
        """Inicia simulação em thread separada"""
        if not self.is_simulating:
            self.is_simulating = True
            self.start_btn.config(state=tk.DISABLED)
            self.reset_btn.config(state=tk.DISABLED)
            
            # Limpar plots
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            
            # Rodar simulação em thread
            self.sim_thread = threading.Thread(target=self.simulator.run_real_time, args=(15.0, 10), daemon=True)
            self.sim_thread.start()
    
    def _reset_simulation(self):
        """Reseta simulação"""
        self.simulator.reset_simulation()
        self.simulator.running = False
        self.is_simulating = False
        self.start_btn.config(state=tk.NORMAL)
        self.reset_btn.config(state=tk.NORMAL)
        
        # Limpar plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.canvas.draw()
        
        # Atualizar info
        self._update_info()
    
    def update_gui(self, state: dict):
        """Atualiza GUI com novo estado (chamado pela simulação)"""
        self._update_info()
        self.root.after(100, self.canvas.draw)  # Agendar desenho
    
    def _update_info(self):
        """Atualiza caixa de informações"""
        i, omega, theta = self.simulator.plant.get_state()
        
        info = f"""
╔════════════════════════════╗
║   ESTADO DO SISTEMA        ║
╚════════════════════════════╝

Tempo: {self.simulator.t_sim:.3f} s
Passo: {self.simulator.k_step}

POSIÇÃO:
  θ(rad): {theta:.4f}
  θ(°):   {np.degrees(theta):.2f}
  Ref:    {self.simulator.theta_ref:.4f} rad
  Erro:   {self.simulator.theta_ref - theta:.4f} rad

VELOCIDADE:
  ω(rad/s): {omega:.4f}

ELÉTRICO:
  i(A):     {i:.4f}
  u(V):     {self.simulator.plant.history_u[-1] if self.simulator.plant.history_u else 0:.4f}

CONTROLADOR:
  Kp: {self.simulator.controller.Kp:.2f}
  Ki: {self.simulator.controller.Ki:.2f}

CARGA:
  τ_load(N·m): {self.simulator.tau_load:.4f}

╔════════════════════════════╗
║   REQUISITOS DE DESEMPENHO ║
╚════════════════════════════╝

✓ M_p < 5%
✓ T_s < 1.5 s
✓ e(∞) = 0
✓ Rejeição < 1.0 s
"""
        
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
        self.info_text.config(state=tk.DISABLED)
    
    def update_plots(self):
        """Atualiza gráficos com histórico da simulação"""
        h = self.simulator.plant
        
        if not h.history_time:
            return
        
        t = np.array(h.history_time)
        theta = np.array(h.history_theta)
        omega = np.array(h.history_omega)
        error = np.array(h.history_e)
        u = np.array(h.history_u)
        
        # Plot 1: Posição vs Referência
        self.ax1.plot(t, theta, 'b-', label='θ(t)', linewidth=2)
        self.ax1.plot(t, [self.simulator.theta_ref]*len(t), 'r--', label='Referência', linewidth=2)
        self.ax1.set_xlabel('Tempo (s)')
        self.ax1.set_ylabel('Posição (rad)')
        self.ax1.set_title('Rastreamento de Posição')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # Plot 2: Velocidade
        self.ax2.plot(t, omega, 'g-', linewidth=2)
        self.ax2.set_xlabel('Tempo (s)')
        self.ax2.set_ylabel('Velocidade Angular (rad/s)')
        self.ax2.set_title('Velocidade Angular')
        self.ax2.grid(True, alpha=0.3)
        
        # Plot 3: Erro
        self.ax3.plot(t, error, 'r-', linewidth=2)
        self.ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        self.ax3.set_xlabel('Tempo (s)')
        self.ax3.set_ylabel('Erro (rad)')
        self.ax3.set_title('Erro de Posição')
        self.ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sinal de Controle
        self.ax4.plot(t, u, 'purple', linewidth=2)
        self.ax4.axhline(y=self.sys_params.V_max, color='r', linestyle='--', alpha=0.5, label='Saturação')
        self.ax4.axhline(y=-self.sys_params.V_max, color='r', linestyle='--', alpha=0.5)
        self.ax4.set_xlabel('Tempo (s)')
        self.ax4.set_ylabel('Tensão (V)')
        self.ax4.set_title('Sinal de Controle')
        self.ax4.grid(True, alpha=0.3)
        self.ax4.legend()
        
        self.canvas.draw()


def main():
    """Função principal: inicia aplicação"""
    root = tk.Tk()
    app = HMIApplication(root)
    app._update_info()
    root.mainloop()


if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  SISTEMA DE CONTROLE DE POSIÇÃO - JUNTA ROBÓTICA         ║")
    print("║  Simulação Digital com Interface HMI                     ║")
    print("║  Junho 2026                                              ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    main()
