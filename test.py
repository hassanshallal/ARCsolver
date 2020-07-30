from inductive_core_knowledge import *
training_path = 'C:\\Users\\12092\\Downloads\\ARCsolver_master\\data\\training'
tt = load_set(training_path)
print(len(tt))

from timeit import default_timer as timer
start = timer()
today_analysis = []
for n in range(len(tt)):
    print(n)
    this_task = Inductive(tt[n])
    today_analysis.append(this_task)

serialize(today_analysis, 'assets/today_analysis')
end = timer()
print("It took this number of seconds: ", end - start)
print(len(today_analysis))
# It took this number of seconds:  2030.154533