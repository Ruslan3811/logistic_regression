import sys

if len(sys.argv) != 2:
    print("Not 2 arguments")
    exit(1)

def find_min(lst):
    min_val = lst[0]
    for val in lst:
        if val < min_val or min_val == None:
            min_val = val
    return min_val

def find_max(lst):
    max_val = lst[0]
    for val in lst:
        if val > max_val or max_val == None:
            max_val = val
    return max_val

def find_std(lst):
    delta = 0
    sum_ms = 0
    meaning = sum(col_dict[labels[ind]]) / (len(col_dict[labels[ind]]) - 1)
    for val in lst:
        delta = pow((val - meaning), 2)
        sum_ms += delta
    return pow(sum_ms / (len(lst) - 1), 0.5)

def find_percentil(lst, perc):
    lst.sort()
    return (lst[int(len(lst) * perc - 1)])

def find_mean(lst):
    return sum(lst) / len(lst)

def count_empty_vals(lst):
    count = 0
    for x in lst:
        print(type(x))
        if x == "":
            count += 1
    return count

# открытие и построчное чтение csv 
with open(sys.argv[1]) as file:
    rows_list = file.read().splitlines()
    col_dict = {}
    labels = rows_list[0].split(',')
    for label in labels:
        col_dict[label] = []
    for row in rows_list[1:]:
        values = row.split(",")
        for index in range(len(col_dict)):
            col_dict[labels[index]].append(values[index])


#удаление ненужных колонок
col_dict.pop("Hogwarts House")
col_dict.pop("First Name")
col_dict.pop("Last Name")
col_dict.pop('Birthday')
col_dict.pop("Best Hand")

# запись названия колонок(фичей) 
labels = [keys for keys in col_dict.keys()]
spaces = 15

count_empty_vals = {}
for label, values in col_dict.items():
    count_empty_vals[label] = 0
    for val in values:
        if val == "":
            count_empty_vals[label] += 1

for label in col_dict.keys():
    col_dict[label] = [float(val) if val != '' else 0 for val in col_dict[label]]


print(" " * spaces, end="")
for label in labels[:8]:
    print(label, end="|  ")
print("\n")
metrics = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Empty"]
ind_metric = 0
for metric in metrics:
    print(metric, end="")
    for ind in range(0, len(col_dict) - 6):
        if metric == "Count":
            metric_nbr = len(col_dict[labels[ind]]) - 1
        elif metric == "Mean":
            metric_nbr = find_mean(col_dict[labels[ind]])
        elif metric == "Std":
            metric_nbr = find_std(col_dict[labels[ind]])
        elif metric == "Min":
            metric_nbr = find_min(col_dict[labels[ind]])
        elif metric == "Max":
            metric_nbr = find_max(col_dict[labels[ind]])
        elif metric == "25%":
            metric_nbr = find_percentil(col_dict[labels[ind]], 0.25)
        elif metric == "50%":
            metric_nbr = find_percentil(col_dict[labels[ind]], 0.5)
        elif metric == "75%":
            metric_nbr = find_percentil(col_dict[labels[ind]], 0.75)
        elif metric == "Empty":
            metric_nbr = count_empty_vals[labels[ind]]
        metric_nbr = format(metric_nbr, '.6f')
        if ind == 0:
            print((len(labels[ind]) + spaces - len(metric) - len(str(metric_nbr))) * " ", end="")
        else:
            print((len(labels[ind]) - len(str(metric_nbr)) + 3) * " ", end="")
        print((metric_nbr), end="")
    ind_metric += 1
    if ind_metric > 0:
        print("")

print("\n")
spaces = 20
print(" " * spaces, end="")
for label in labels[8:]:
    print(label, end="|      ")
print("\n")
ind_metric = 0
for metric in metrics:
    print(metric, end="")
    for ind in range(8, len(col_dict)):
        if metric == "Count":
            metric_nbr = len(col_dict[labels[ind]]) - 1
        elif metric == "Mean":
            metric_nbr = sum(col_dict[labels[ind]]) / (len(col_dict[labels[ind]]) - 1)
        elif metric == "Std":
            metric_nbr = find_std(col_dict[labels[ind]])
        elif metric == "Min":
            metric_nbr = find_min(col_dict[labels[ind]])
        elif metric == "Max":
            metric_nbr = find_max(col_dict[labels[ind]])
        elif metric == "25%":
            metric_nbr = find_percentil(col_dict[labels[ind]], 0.25)
        elif metric == "50%":
            metric_nbr = find_percentil(col_dict[labels[ind]], 0.5)
        elif metric == "75%":
            metric_nbr = find_percentil(col_dict[labels[ind]], 0.75)
        elif metric == "Empty":
            metric_nbr = count_empty_vals[labels[ind]]
        metric_nbr = format(metric_nbr, '.6f')
        if ind == 8:
            print((len(labels[ind]) + spaces - len(metric) - len(str(metric_nbr))) * " ", end="")
        else:
            print((len(labels[ind]) - len(str(metric_nbr)) + 7) * " ", end="")
        print((metric_nbr), end="")
    ind_metric += 1
    if ind_metric > 0:
        print("")

print("\n")