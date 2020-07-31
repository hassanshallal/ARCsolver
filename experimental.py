from inductive_core_knowledge import *
# training_path = 'C:\\Users\\12092\\Desktop\\ARCsolver\\data\\training'
training_path = '/home/hshallal/Desktop/ARCsolver/data/training'
tt = load_set(training_path)
print(len(tt))

knowns = load_serialized('assets/solved_indices_for_the_day.pkl')
previous_analysis = load_serialized('assets/today_analysis')

def merge_sub_objectives(sub_test, sub_lead):
    if len(sub_test[0]) == 2 and len(sub_test) > 2 and len(sub_lead) > 2 and sub_test[0:2] == sub_lead[0:2]:
        intersect = sorted(list(set(sub_test[2]).intersection(set(sub_lead[2]))))
        merger = [sub_lead[0], sub_lead[1], intersect]
        for n in intersect:
            in_sub_test = [x for x in sub_test if x[0] == n and type(x) == tuple]
            in_sub_lead = [x for x in sub_lead if x[0] == n and type(x) == tuple]
            new_sit = (n, in_sub_test[0][1].union(in_sub_lead[0][1]), in_sub_test[0][2].union(in_sub_lead[0][2]))
            merger.append(new_sit)
        return merger
    elif len(sub_test[0]) == 3 and  len(sub_test) > 1 and len(sub_lead) > 1 and sub_test[0] == sub_lead[0]:
        intersect = sorted(list(set(sub_test[1]).intersection(set(sub_lead[1]))))
        merger = [sub_lead[0], intersect]
        for n in intersect:
            in_sub_test = [x for x in sub_test if x[0] == n and type(x) == tuple]
            in_sub_lead = [x for x in sub_lead if x[0] == n and type(x) == tuple]
            new_sit = (n, in_sub_test[0][1].union(in_sub_lead[0][1]))
            merger.append(new_sit)
        return merger
    elif len(sub_test[0]) == 3 and  len(sub_test) > 1 and len(sub_lead) > 1 and sub_test[0] != sub_lead[0]:
        intersect = sorted(list(set(sub_test[1]).intersection(set(sub_lead[1]))))
        common = [x for x in sub_test if x in sub_lead]
        if len(common) > 0:
            intersect = [x for x in intersect if x in [y[0] for y in common]]
            merger = [sub_test[0], sub_lead[0], intersect] + common
            return merger
    elif len(sub_test[0]) == 3 and len(sub_test) == 1 and sub_test[0] == sub_lead[0]:
        return sub_lead
    else:
        return sub_lead

def is_complete_sub_objective(sub_objective):
            return len(sub_objective) != 2
    
def validate_running_objective(this_task):
    len_list = [len(x) for x in this_task.running_objective]
    lead_index = len_list.index(max(len_list))
    
    lead_objective = this_task.running_objective[lead_index]
    for n in range(len(this_task.running_objective)):       
        if n != lead_index:
            current_sub = this_task.running_objective[n]
            for x in range(len(current_sub)):
                for y in range(len(lead_objective)):
#                     print('current_sub[x]: ', current_sub[x])
#                     print('lead_objective[y]: ', lead_objective[y])
                    lead_objective[y] = merge_sub_objectives(current_sub[x], lead_objective[y])
#                     print('New:', lead_objective[y])
                    
    return all([is_complete_sub_objective(x) for x in lead_objective]), lead_objective
        
#plot_task(tt[193]) # consistent because it is just mu;ltiplied and everything stays, good observation
#print(previous_analysis[193].solved, previous_analysis[193].mechanisms)
unsolved = [1, 119, 186, 250, 193, 337, 345, 99, 128, 338]
for n in unsolved:
    print(n)
    print(previous_analysis[n].objective_status)
    print(previous_analysis[n].running_objective)
    print(validate_running_objective(previous_analysis[n]))
    print('=====')

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

