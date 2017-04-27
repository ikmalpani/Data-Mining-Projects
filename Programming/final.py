import sys	
import re
import operator
from collections import Counter, OrderedDict

def read_input(input_data,parameter_data,output_patterns):
	with open(input_data,"r") as f1:
		trans = f1.read().splitlines()
		i=0
		for each in trans:
			trans[i] = trans[i].replace('{', '')
			trans[i] = trans[i].replace('}', '')
			trans[i] = trans[i].split(', ')
			i=i+1
		len_trans = i;
		#print(trans)

	with open(parameter_data,"r") as f2:
		parameters = []
		parameters = f2.read().splitlines()
		mis = parameters[0:-3]
		data = [row for row in mis]
		i=0;
		minSupDict = {}
		MIS = {}
		for i in range(len(data)):
			items = list(map(lambda x:x.strip(),data[i].split("=")))
			reSearch = re.search(r'MIS\(?(\w+)\)?',items[0])
			key = reSearch.group(1)
			minSupDict[tuple([key])] = float(items[1])
			value = minSupDict[tuple([key])]
			new = {key:value}
			MIS.update(new)
		#print(MIS)

		must_have = parameters[-1]
		items1 = must_have.strip()
		items1 = items1.split(": ")
		must_have = items1[1].split(" or ")
		#print(must_have)
		
		sdc = parameters[-3]
		items2 = sdc.strip()
		items2 = items2.split(" = ")
		sdc = items2[1]
		#print(sdc)
		
		cannot_be = parameters[-2]
		items3 = cannot_be.strip()
		items3 = items3.split(": ")
		cannot_be = list(map(lambda x : str(x),list(eval((items3[1])))))
		i = 0
		for each in cannot_be:
			cannot_be[i] = cannot_be[i].replace('{', '')
			cannot_be[i] = cannot_be[i].replace('}', '')
			cannot_be[i] = cannot_be[i].split(', ')
			i=i+1
		#print(cannot_be)
	return trans, MIS, must_have, sdc, cannot_be, len_trans

def getL(trans,MIS,sorted_MIS):
	list1 = [x[1] for x in sorted_MIS]
	L = {}
	i = 0
	temp4 = []
	temp5 = []
	#print(list1)
	temp_list = []
	for each in range(len(trans)):
		#print(trans[i])
		j=0
		for each in range(len(trans[i])):
			temp_list.append(trans[i][j])
			counter = dict(Counter(temp_list))
			j = j + 1
		i = i + 1
	new_dict ={}
	temp_list2 =[]
	temp_list6 = []
	temp_list7 = []
	for each in counter:
		value = counter.get(each) 
		#print(value)
		sup_count = value/i 
		temp_list2.append(sup_count) 
		temp_list3 = list(counter.keys()) 
	new_dict = dict(zip(temp_list3,temp_list2))
	#print(temp_list2)
	#print(new_dict)
	first = sorted_MIS[0][1]
	for each in new_dict:
		if(first<new_dict[each]):
			temp_list7.append(each)
			temp_list6.append(new_dict[each])
	L1 = dict(zip(temp_list7,temp_list6))
	return L1

def special_conditions(L,trans,must_have,cannot_be,MIS):
	F1 = []
	temp4 = []
	temp5 = []
	temp = []
	k = 2

	for each in L:
		if MIS[each] <= L[each]:
			temp4.append(each)
			temp5.append(L[each])
	L2 = dict(zip(temp4,temp5))

	for each in L2:
		if each in must_have:
			F1.append(each)
			temp.append(L2[each])
	L3 = F1
	counter = 0
	count = []
	i = len(F1)
	j = 0
	return L2,L3,count

def special_conditions_k(L,trans,must_have,cannot_be):
	for i in range(1, len(L)):
		L[i] = list(L[i])
	L_temp = []
	i = 0
	for each in L:
		temp = L[i]

		for each in cannot_be:
			if set(temp) & set(each) == set(each):
				L_temp.append(temp)
				i = i - 1
		i = i +1
	L = [each for each in L if each not in L_temp]	
	L = [any_in for any_in in L if len(set(any_in).intersection(must_have))>=1]
	return L

def level2_cangen(sorted_MIS,MIS,trans,sdc,L):
	C2 = []
	count = {}
	Fk = []
	tail_count = {}
	C3 = []
	F = {}
	for i in range(0,len(MIS)-1):
		if sorted_MIS[i][0] in L.keys() and L[sorted_MIS[i][0]] >= MIS[sorted_MIS[i][0]]:
			for j in range(i+1,len(sorted_MIS)):
				if sorted_MIS[j][0] in L.keys() and L[sorted_MIS[j][0]] >= MIS[sorted_MIS[i][0]] and abs(L[sorted_MIS[j][0]] - L[sorted_MIS[i][0]])  <= float(sdc):
					temp_list = []
					temp_list.append(sorted_MIS[i][0])
					temp_list.append(sorted_MIS[j][0])
					C2.append(tuple(temp_list))
	C2 = [[i] for i in C2]
	C2 = [i[0] for i in C2]
	for c in C2:
		d = tuple(sorted(c, key = lambda i : MIS[i]))
		count[d] = 0
		tail_count[d] = 0

	for t in trans:
		for c in C2:
			if set(t) & set(c) == set(c):
				d = tuple(sorted(c, key = lambda i : MIS[i]))
				count[d] += 1
				tail_count[d] += 1
			if set(t) & set(c[1:]) == set(c[1:]):    # tail count
				tail = c[1:]
				A = list(tail_count.keys())
				A = [[i] for i in A]
				for each in A:
					if tail in each:
						tail_count[tail] += 1
					else:
						tail_count[tail] = 1
	for c in C2:
		d = tuple(sorted(c, key = lambda i : MIS[i]))
		if count[d] >= len(trans) * MIS[ c[0] ]:
			Fk.append(c)
	return C2,Fk,count, tail_count


def MS_cangen(Fk_1, sdc,trans,MIS):
    Ck = []
    Fk = []
    Fk_1 = [list(x) for x in Fk_1]
    for i in range(len(Fk_1) - 1):
        f1 = Fk_1[i]
        f1.sort()
        for j in range(i+1, len(Fk_1)):    
            f2 = Fk_1[j]
            f2.sort()

            if f1[:len(f1)-1] == f2[:len(f2)-1] and f1[-1] < f2[-1]:
            	sup1 = float(sub( trans, [f1[-1]] )) / len(trans)
            	sup2 = float(sub( trans, [f2[-1]] )) / len(trans)
            	if abs(sup1 - sup2) <= float(sdc):
                	f1.append(f2[-1])
                	c = f1
                	Ck.append(c)
    count = {}
    tail_count = {}
    for c in Ck:
    	d = tuple(sorted(c, key = lambda i : MIS[i]))
    	count[d] = 0
    	tail_count[d] = 0
    for t in trans:
    	for c in Ck:
    		if set(t) & set(c) == set(c):
    			d = tuple(sorted(c, key = lambda i : MIS[i]))
    			count[d] += 1
    			tail_count[d] += 1

    		if set(t) & set(c[1:]) == set(c[1:]):    # tail count
    			tail = tuple(c[1:])
    			#print(type(tail))
    			if tail in tail_count.keys():
    				#print('yes')
    				tail_count[tail] += 1
    			else:
    				tail_count[tail] = 1
    for c in Ck:
    	d = tuple(sorted(c, key = lambda i : MIS[i]))
    	if count[d] >= len(trans) * MIS[ c[0] ]:
    		Fk.append(c)
    return Ck,Fk,count, tail_count

def sub(trans, item):
    count = 0
    for i in range(len(trans)):
        if set(trans[i]) & set(item) == set(item):
            count += 1
    return count


def __main__(argv):
	input_data = argv[1]
	parameter_data = argv[2]
	output_patterns = argv[3]

	trans, MIS, must_have, sdc, cannot_be, len_trans = read_input(input_data,parameter_data,output_patterns)
	sorted_MIS = sorted(MIS.items(), key=operator.itemgetter(1))
	L = getL(trans,MIS,sorted_MIS)
	L1,F1,counter = special_conditions(L,trans,must_have,cannot_be,MIS)
	counter = 0
	count = []
	i = len(F1)
	j = 0
	write = ""
	with open(output_patterns,"w") as f:
		print('Frequent 1- itemsets:')
		write += 'Frequent 1 - itemsets: \n'
		for each in F1:
			counter = 0
			for each in trans:
				if each.count(F1[j]):
					counter = counter + 1
			count.append(counter)
			print('\t',count[j], ': {',(F1[j]),'}')
			write += '\t'+str(count[j])+ ': {'+str((F1[j]))+'}\n\n'
			j = j + 1
		finalSet = []
		summary = OrderedDict()
		Fk = L1
		k = 2
		print('Total number of Frequent 1 sets:',len(F1)) 
		write += 'Total number of Frequent 1 sets: '+str(len(F1))+'\n\n'
		while len(Fk) != 0:
			if k == 2:
				Ck,Fk,count,tail_count = level2_cangen(sorted_MIS,MIS,trans,sdc,L)
				final = special_conditions_k(Fk,trans,must_have,cannot_be)
				if len(final) != 0:
					print("Frequent %d-itemsets: \n" %k)
					write += "Frequent "+ str(k) +"-itemsets: \n"
					i=0
					j=0
					l=0
					for i in final:
						for j in count:
							if tuple(i) == j:
								#print(i)
								print ('\t',count[j],': {',*i,' }')
								write += '\t'+str(count[j])+': {'+ str(i) + '}\n'
						for l in tail_count:
							if tuple(i) == l:
								print('Tail count : ',tail_count[l])
								write += 'Tail count : '+str(tail_count[l])+'\n'
					print('\nTotal number of frequent 2 itemsets: ',len(final))
					write += '\nTotal number of frequent 2 itemsets: ' + str(len(final))+'\n\n'
				k = k + 1
			else:
				Ck,Fk,count,tail_count = MS_cangen(Fk,sdc,trans,MIS)
				final = special_conditions_k(Fk,trans,must_have,cannot_be)
				if len(final)!=0:
					print("Frequent %d-itemsets:\n" %k)
					write += "Frequent "+str(k)+"-itemsets: \n" 
					i=0
					j=0
					l=0
					for i in final:
						for j in count:
							if tuple(i) == j:
								print ('\t',count[j],':{',*i,' }')
								write += '\t'+str(count[j])+':'+str(i)+'\n'
						for l in tail_count:
							if tuple(i) == l:
								print('Tail count : ',tail_count[l])
								write +='Tail count : '+str(tail_count[l])+'\n'
					print('\nTotal number of frequent itemsets: ',len(final))
					write +='\nTotal number of frequent '+str(k)+   ' itemsets: '+str(len(final))+'\n\n'
				k = k + 1
		f.write(write)

if __name__ == "__main__":
	__main__(sys.argv)