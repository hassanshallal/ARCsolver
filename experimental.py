from inductive_core_knowledge import *
training_path = 'C:\\Users\\12092\\Downloads\\ARCsolver_master\\data\\training'
tt = load_set(training_path)
print(len(tt))

knowns = load_serialized('assets/solved_indices_for_the_day.pkl')
previous_analysis = load_serialized('assets/today_analysis')

# we ned to fix the consistency
def test_consistency(this_task):

    len_list = [len(x) for x in this_task.running_objective]
    lead_index = len_list.index(max(len_list))
    lead_objective = this_task.running_objective[lead_index]
    print('lead_objective: ', lead_objective)
    consistency_measure = [None] * len(this_task.running_objective)
    consistency_measure[lead_index] = True
    print(consistency_measure)
    for n in range(len(this_task.running_objective)):
        if n != lead_index:
            print('test_objective: ', this_task.running_objective[n])
            this_case_consistency = [None] * len(this_task.running_objective[n])
            print('this_case_consistency: ', this_case_consistency)
            for m in range(len(this_task.running_objective[n])):
                if this_task.running_objective[n][m] not in lead_objective: # replace with a less stringent condition and modify your lead objective on the fly
                    this_case_consistency[m] = False
                else:
                    this_case_consistency[m] = True
                print('this_case_consistency: ', this_case_consistency)
            consistency_measure[n] = all(this_case_consistency)
            
        print(consistency_measure)

    return all(consistency_measure), lead_objective

    

plot_task(tt[193]) # consistent because it is just mu;ltiplied and everything stays, good observation
print(previous_analysis[193].solved, previous_analysis[193].mechanisms)
# unsolved = [1, 119, 186, 250, 193, 337, 345, 99, 128, 338]
# for n in unsolved:
#     print(n)
#     print(previous_analysis[n].objective_status)
#     print(previous_analysis[n].running_objective)
#     print(test_consistency(previous_analysis[n]))
#     print('=====')

#solved = 0
#    print(n)
#    plot_task(tt[n])
#    this_task = Inductive(tt[n])
#if previous_analysis[n].solved != 'solved':
#  print(n)
#        solved += 1
#        print(this_task.mechanisms)
#        print(this_task.running_objective)
#    else:
#        print(this_task.running_objective)
#    print('======')

