import seaborn as sns
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import time
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
from parameters import Parameters
from leaf_temperature import Environment, ODE

def model(params):
    p = Parameters()
    p.gb = params[0]
    p.k = params[1]
    p.Vcmax25 = params[2]
    p.Jmax25 = params[3]
    p.PLDENS=params[4]
    p.SCV=params[5]
    p.g0=params[6]
    p.g1=params[7]
    p.Rd25=params[8]
    p.KDIF=params[9]
    p.Ds0=params[10]
    p.WLVE = params[11]
    p.tauAd=params[12]
    p.tauAi = params[13]
    p.tauGd = params[14]
    p.tauGi = params[15]
    p.Abs = params[16]
    p.Curv = params[17]
    p.gm25 = params[18]
    print(f"Parameter values:{params}")

    env = Environment()
    y0 = [env.Tair[0], -p.Rd25, p.g0]
    print(f"y0 is:{y0}")
    t_span = (0, 3 * 24 * 60 * 60)
    t_points = np.arange(t_span[0], t_span[1] + 1, 600)

    start_time = time.time()
    try:
        solution = solve_ivp(ODE,
                             t_span,
                             y0,
                             method='BDF',
                             t_eval=t_points,
                             dense_output=False,
                             vectorized=True,
                             args=(env, p)
                             )
    except ValueError as e:
        print(f"ValueError encountered: {e}")
        print(f"Parameters: {params}")
        return np.nan

    solver_time = time.time() - start_time

    print(f'Time taken by the ODE solver in model function: {solver_time:.4f} seconds')
    if solution.success:

        leaf_temp = solution.y[0].mean()
        A = solution.y[1].mean()
        gs = solution.y[2].mean()

        print(f'Final results - leaf_temp: {leaf_temp}, A: {A}, gs: {gs}')
        return leaf_temp, A, gs
    else:
        print("ODE solver failed.")
        return np.nan, np.nan, np.nan

def sensitivity_analysis(Y, problem):
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=True, conf_level=0.95, print_to_console=False)
    return Si

# Test
p = Parameters()
env = Environment()

problem = {
    'num_vars': 19,
    'names': ['gb', 'k', 'Vcmax25', 'Jmax25','PLDENS','SCV','g0','g1','Rd25',
              'KDIF','Ds0','WLVE','tauAd','tauAi','tauGd','tauGi','Abs','Curv','gm25'],
    'bounds': [
        [0.05, 2.5],    # gb
        [450, 1200],    # k
        [50.0, 200],      # Vcmax25
        [100, 250],     # Jmax25
        [1.20, 5.0],     # PLDENS
        [0.1, 0.3],     # SCV
        [0.01, 0.1],    # g0
        [5.0, 15.0],    # g1
        [0.5, 2.0],     # Rd25
        [0.3, 1.0],     # KDIF
        [0.5, 2.0],     # Ds0
        [5.0, 20.0],      #WLVE
        [0.5, 5.0],      #tauAd
        [60, 600],       #tauAi
        [150, 1000],    #tauGd
        [150, 1000],   #tauGi
        [0.4, 1.0],    #Abs
        [0.4, 1.0],    #Curv
        [1.0, 5.0]     # gm25
    ]
}

param_values = sobol.sample(problem, 128, calc_second_order=True)

# Ntotal = N*(2D+2) where D indicates the number of parameter. N is multiples of 2: 4,8,16...

Y = []
for params in param_values:
    result = model(params)
    print(f"for params {params}, model output is {result}")
    Y.append(result)

Y = np.array(Y)
Y = np.where(np.isnan(Y), 0, Y)

Si_leaf_temp = sensitivity_analysis(Y[:, 0], problem)
Si_A = sensitivity_analysis(Y[:, 1], problem)
Si_gs = sensitivity_analysis(Y[:, 2], problem)
Si = [Si_leaf_temp,Si_A,Si_gs]

# plot
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'axes.titlesize': 18})
plt.rcParams.update({'axes.labelsize': 16})
plt.rcParams.update({'xtick.labelsize': 14})
plt.rcParams.update({'ytick.labelsize': 14})

label = ['Tleaf','A','gs']
for i in range(len(Si)):
    plt.figure(figsize=(10, 8))
    sns.heatmap(Si[i]['S2'], annot=True, xticklabels=problem['names'], yticklabels=problem['names'], cmap='viridis')
    plt.title(f'Sensitivity Analysis for {label[i]} - Second-order Sobol indices')
    plt.xlabel('Parameter')
    plt.ylabel('Second-order Sobol index (S2)')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()

    sorted_indices_S1 = np.argsort(Si[i]['S1'])[::-1]
    sorted_names_S1 = np.array(problem['names'])[sorted_indices_S1]
    sorted_S1 = np.array(Si[i]['S1'])[sorted_indices_S1]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_names_S1, sorted_S1)
    plt.xlabel('Parameter')
    plt.ylabel('First-order Sobol index (S1)')
    plt.title(f'Sensitivity Analysis for {label[i]} - First-order Sobol indices')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()

    sorted_indices_ST = np.argsort(Si[i]['ST'])[::-1]
    sorted_names_ST = np.array(problem['names'])[sorted_indices_ST]
    sorted_ST = np.array(Si[i]['ST'])[sorted_indices_ST]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_names_ST, sorted_ST)
    plt.xlabel('Parameter')
    plt.ylabel('Total-order Sobol index (ST)')
    plt.title(f'Sensitivity Analysis for {label[i]} - Total-order Sobol indices')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()

# Sensitivity analysis of Tleaf, A, GS in one figure.
names = problem['names']
bar_width = 0.25
index = np.arange(len(names))

plt.figure(figsize=(12, 8))
plt.bar(index - bar_width, Si_leaf_temp['S1'], bar_width, alpha=0.7, label='Tleaf', color='blue')
plt.bar(index, Si_A['S1'], bar_width, alpha=0.7, label='A', color='orange')
plt.bar(index + bar_width, Si_gs['S1'], bar_width, alpha=0.7, label='gs', color='green')

plt.xlabel('Parameters')
plt.ylabel('First-order Sobol index (S1)')
plt.title('Sensitivity Analysis for Tleaf, A, and gs')
plt.xticks(index, [f'{name}' for name in names])
plt.legend()
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.bar(index - bar_width, Si_leaf_temp['ST'], bar_width, alpha=0.7, label='Tleaf', color='blue')
plt.bar(index, Si_A['ST'], bar_width, alpha=0.7, label='A', color='orange')
plt.bar(index + bar_width, Si_gs['ST'], bar_width, alpha=0.7, label='gs', color='green')

plt.xlabel('Parameters')
plt.ylabel('Total-order Sobol index (ST)')
plt.title('Sensitivity Analysis for Tleaf, A, and gs')
plt.xticks(index, [f'{name}' for name in names])
plt.legend()
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# Output storage
# store Si
def save_sensitivity_results(Si_leaf_temp, Si_A, Si_gs, filename='bottom[1]_sobol_indices_16P_3output.npz'):
    np.savez(filename,
             Si_leaf_temp_S1=Si_leaf_temp['S1'],
             Si_leaf_temp_S2=Si_leaf_temp['S2'],
             Si_leaf_temp_ST=Si_leaf_temp['ST'],
             Si_A_S1=Si_A['S1'],
             Si_A_S2=Si_A['S2'],
             Si_A_ST=Si_A['ST'],
             Si_gs_S1=Si_gs['S1'],
             Si_gs_S2=Si_gs['S2'],
             Si_gs_ST=Si_gs['ST'],

             parameter_names=np.array(problem['names']))
    print(f'Sensitivity results saved to {filename}')

save_sensitivity_results(Si_leaf_temp, Si_A, Si_gs, 'bottom[1]_sobol_indices_19P_3output.npz')

# store Y
np.savez('B19Y[1].npz', Y_leaf_temp=Y[:, 0], Y_A=Y[:, 1],Y_gs=Y[:, 2])

# store problem dict
#import json
#with open ('Problem_definition_16parameters.json','w') as f:
#     problem=json.dump(problem, f)

"""
import seaborn as sns
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import time
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
from parameters import Parameters
from leaf_temperature import Environment, ODE

def model(params):
    p = Parameters()
    p.k = params[0]
    p.Vcmax25 = params[1]
    p.Jmax25 = params[2]
    p.PLDENS=params[3]
    p.SCV=params[4]
    p.WLVE = params[5]
    p.tauAd=params[6]
    p.tauAi = params[7]
    p.tauGd = params[8]
    p.tauGi = params[9]
    p.Curv = params[10]
    p.gm25 = params[11]
    print(f"Parameter values:{params}")

    env = Environment()
    y0 = [env.Tair[0], -p.Rd25, p.g0]
    print(f"y0 is:{y0}")
    t_span = (0, 3 * 24 * 60 * 60)
    t_points = np.arange(t_span[0], t_span[1] + 1, 600)

    start_time = time.time()
    try:
        solution = solve_ivp(ODE,  # Rreplace your ODE function name.
                             t_span,
                             y0,
                             method='BDF',
                             t_eval=t_points,
                             dense_output=False,
                             vectorized=True,
                             args=(env, p)
                             )
    except ValueError as e:
        print(f"ValueError encountered: {e}")
        print(f"Parameters: {params}")
        return np.nan

    solver_time = time.time() - start_time

    print(f'Time taken by the ODE solver in model function: {solver_time:.4f} seconds')
    if solution.success:

        leaf_temp = solution.y[0].mean()
        A = solution.y[1].mean()
        gs = solution.y[2].mean()

        print(f'Final results - leaf_temp: {leaf_temp}, A: {A}, gs: {gs}')
        return leaf_temp, A, gs
    else:
        print("ODE solver failed.")
        return np.nan, np.nan, np.nan

def sensitivity_analysis(Y, problem):
    Si = sobol_analyze.analyze(problem, Y, calc_second_order=True, conf_level=0.95, print_to_console=False)
    return Si

# Test
p = Parameters()
env = Environment()

problem = {
    'num_vars': 12,
    'names': ['k', 'Vcmax25', 'Jmax25','PLDENS','SCV','WLVE','tauAd','tauAi','tauGd','tauGi','Curv','gm25'],
    'bounds': [
        [450, 1200],    # k
        [50.0, 200],      # Vcmax25
        [100, 250],     # Jmax25
        [1.20, 5.0],     # PLDENS
        [0.1, 0.3],     # SCV
        [5.0, 20.0],      #WLVE
        [0.5, 5.0],    #tauAd
        [60, 600],    #tauAi
        [150, 1000],    #tauGd
        [150, 1000],   #tauGi
        [0.4, 1.0],    #Curv
        [1.0, 5.0]  # gm25
    ]
}

param_values = sobol.sample(problem, 128, calc_second_order=True)

# Ntotal = N*(2D+2) where D indicates the number of parameter. N is multiples of 2: 4,8,16...

#Y = np.array([model(params) for params in param_values])
#Y = np.where(np.isnan(Y), 0, Y)

Y = []
for params in param_values:
    result = model(params)
    print(f"for params {params}, model output is {result}")
    Y.append(result)

Y = np.array(Y)
Y = np.where(np.isnan(Y), 0, Y)

Si_leaf_temp = sensitivity_analysis(Y[:, 0], problem)
Si_A = sensitivity_analysis(Y[:, 1], problem)
Si_gs = sensitivity_analysis(Y[:, 2], problem)
Si = [Si_leaf_temp,Si_A,Si_gs]

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'axes.titlesize': 18})
plt.rcParams.update({'axes.labelsize': 16})
plt.rcParams.update({'xtick.labelsize': 14})
plt.rcParams.update({'ytick.labelsize': 14})

label = ['Tleaf','A','gs']
for i in range(len(Si)):
    plt.figure(figsize=(10, 8))
    sns.heatmap(Si[i]['S2'], annot=True, xticklabels=problem['names'], yticklabels=problem['names'], cmap='viridis')
    plt.title(f'Sensitivity Analysis for {label[i]} - Second-order Sobol indices')
    plt.xlabel('Parameter')
    plt.ylabel('Second-order Sobol index (S2)')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()

    sorted_indices_S1 = np.argsort(Si[i]['S1'])[::-1]
    sorted_names_S1 = np.array(problem['names'])[sorted_indices_S1]
    sorted_S1 = np.array(Si[i]['S1'])[sorted_indices_S1]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_names_S1, sorted_S1)
    plt.xlabel('Parameter')
    plt.ylabel('First-order Sobol index (S1)')
    plt.title(f'Sensitivity Analysis for {label[i]} - First-order Sobol indices')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()

    sorted_indices_ST = np.argsort(Si[i]['ST'])[::-1]
    sorted_names_ST = np.array(problem['names'])[sorted_indices_ST]
    sorted_ST = np.array(Si[i]['ST'])[sorted_indices_ST]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_names_ST, sorted_ST)
    plt.xlabel('Parameter')
    plt.ylabel('Total-order Sobol index (ST)')
    plt.title(f'Sensitivity Analysis for {label[i]} - Total-order Sobol indices')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()


# Sensitivity analysis of Tleaf, A, GS in one figure.
names = problem['names']
bar_width = 0.25
index = np.arange(len(names))

plt.figure(figsize=(12, 8))
plt.bar(index - bar_width, Si_leaf_temp['S1'], bar_width, alpha=0.7, label='Tleaf', color='blue')
plt.bar(index, Si_A['S1'], bar_width, alpha=0.7, label='A', color='orange')
plt.bar(index + bar_width, Si_gs['S1'], bar_width, alpha=0.7, label='gs', color='green')

plt.xlabel('Parameters')
plt.ylabel('First-order Sobol index (S1)')
plt.title('Sensitivity Analysis for Tleaf, A, and gs')
plt.xticks(index, [f'{name}' for name in names])
plt.legend()
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.bar(index - bar_width, Si_leaf_temp['ST'], bar_width, alpha=0.7, label='Tleaf', color='blue')
plt.bar(index, Si_A['ST'], bar_width, alpha=0.7, label='A', color='orange')
plt.bar(index + bar_width, Si_gs['ST'], bar_width, alpha=0.7, label='gs', color='green')

plt.xlabel('Parameters')
plt.ylabel('Total-order Sobol index (ST)')
plt.title('Sensitivity Analysis for Tleaf, A, and gs')
plt.xticks(index, [f'{name}' for name in names])
plt.legend()
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# Output storage
# store Si
def save_sensitivity_results(Si_leaf_temp, Si_A, Si_gs, filename='Top_sobol_indices_12P_3output.npz'):
    np.savez(filename,
             Si_leaf_temp_S1=Si_leaf_temp['S1'],
             Si_leaf_temp_S2=Si_leaf_temp['S2'],
             Si_leaf_temp_ST=Si_leaf_temp['ST'],
             Si_A_S1=Si_A['S1'],
             Si_A_S2=Si_A['S2'],
             Si_A_ST=Si_A['ST'],
             Si_gs_S1=Si_gs['S1'],
             Si_gs_S2=Si_gs['S2'],
             Si_gs_ST=Si_gs['ST'],

             parameter_names=np.array(problem['names']))
    print(f'Sensitivity results saved to {filename}')

save_sensitivity_results(Si_leaf_temp, Si_A, Si_gs, 'Top_sobol_indices_12P_3output.npz')

# store Y
np.savez('Y.npz', Y_leaf_temp=Y[:, 0], Y_A=Y[:, 1],Y_gs=Y[:, 2])

# store problem dict
import json
with open ('Problem_definition_12parameters.json','w') as f:
     problem=json.dump(problem, f)
"""