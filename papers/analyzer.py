import sys
sys.path.insert(0, '../ELINA/python_interface/')
from collections import defaultdict
import pdb
import logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
import numpy as np
import re
import csv
from elina_box import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from elina_scalar import *
from elina_interval import *
from elina_linexpr0 import *
from elina_lincons0 import *
import ctypes
from ctypes.util import find_library
from gurobipy import *
import time

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, 'stdout')
printf = libc.printf

class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.numlayer = 0
        self.ffn_counter = 0

def parse_bias(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    #return v.reshape((v.size,1))
    return v

def parse_vector(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    return v.reshape((v.size,1))
    #return v

def balanced_split(text):
    i = 0
    bal = 0
    start = 0
    result = []
    while i < len(text):
        if text[i] == '[':
            bal += 1
        elif text[i] == ']':
            bal -= 1
        elif text[i] == ',' and bal == 0:
            result.append(text[start:i])
            start = i+1
        i += 1
    if start < i:
        result.append(text[start:i])
    return result

def parse_matrix(text):
    i = 0
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    return np.array([*map(lambda x: parse_vector(x.strip()).flatten(), balanced_split(text[1:-1]))])

def parse_net(text):
    lines = [*filter(lambda x: len(x) != 0, text.split('\n'))]
    i = 0
    res = layers()
    while i < len(lines):
        if lines[i] in ['ReLU', 'Affine']:
            W = parse_matrix(lines[i+1])
            b = parse_bias(lines[i+2])
            res.layertypes.append(lines[i])
            res.weights.append(W)
            res.biases.append(b)
            res.numlayer+= 1
            i += 3
        else:
            raise Exception('parse error: '+lines[i])
    return res
   
def parse_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    with open('dummy', 'w') as my_file:
        my_file.write(text)
    data = np.genfromtxt('dummy', delimiter=',',dtype=np.double)
    low = np.copy(data[:,0])
    high = np.copy(data[:,1])
    return low,high

def get_perturbed_image(x, epsilon):
    image = x[1:len(x)]
    num_pixels = len(image)
    LB_N0 = image - epsilon
    UB_N0 = image + epsilon
     
    for i in range(num_pixels):
        if(LB_N0[i] < 0):
            LB_N0[i] = 0
        if(UB_N0[i] > 1):
            UB_N0[i] = 1
    return LB_N0, UB_N0


def generate_linexpr0(weights, bias, size):
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_DENSE, size)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, bias)
    for i in range(size):
        elina_linexpr0_set_coeff_scalar_double(linexpr0,i,weights[i])
    return linexpr0


def prerun (nn, LB_N0, UB_N0, label):
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer 
    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(num_pixels)
    for i in range(num_pixels):
        elina_interval_set_double(itv[i],LB_N0[i],UB_N0[i])

    ## construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_pixels, itv)
    elina_interval_array_free(itv,num_pixels)
    for layerno in range(numlayer):
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
           weights = nn.weights[nn.ffn_counter]
           biases = nn.biases[nn.ffn_counter]
           dims = elina_abstract0_dimension(man,element)
           num_in_pixels = dims.intdim + dims.realdim
           num_out_pixels = len(weights)
           dimadd = elina_dimchange_alloc(0,num_out_pixels)    
           for i in range(num_out_pixels):
               dimadd.contents.dim[i] = num_in_pixels
           elina_abstract0_add_dimensions(man, True, element, dimadd, False)
           elina_dimchange_free(dimadd)
           np.ascontiguousarray(weights, dtype=np.double)
           np.ascontiguousarray(biases, dtype=np.double)
           var = num_in_pixels

           # handle affine layer

           for i in range(num_out_pixels):
               tdim= ElinaDim(var)
               linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
               element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
               var+=1

           dimrem = elina_dimchange_alloc(0,num_in_pixels)
           for i in range(num_in_pixels):
               dimrem.contents.dim[i] = i
           elina_abstract0_remove_dimensions(man, True, element, dimrem)
           elina_dimchange_free(dimrem)
           # handle ReLU layer 
           if(nn.layertypes[layerno]=='ReLU'):
              element = relu_box_layerwise(man,True,element,0, num_out_pixels)
           nn.ffn_counter+=1 

        else:
           print(' net type not supported')

    dims = elina_abstract0_dimension(man,element)
    output_size = dims.intdim + dims.realdim
    # get bounds for each output neuron
    bounds = elina_abstract0_to_box(man,element)
    
    verified_flag = True
    predicted_label = 0
    label_dict = defaultdict(float)

    for i in range(output_size):
        sup = bounds[i].contents.sup.contents.val.dbl
        label_dict[i] = sup

    for i in range(output_size):
        inf = bounds[i].contents.inf.contents.val.dbl
        flag = True
        for j in range(output_size):
            if(j!=i):
                sup = bounds[j].contents.sup.contents.val.dbl
                if(inf<=sup):
                    flag = False
                    break
        if(flag):
            predicted_label = i
            break   

    label_dict = sorted(label_dict.items(), key=lambda kv: kv[1], reverse=True) 

    elina_interval_array_free(bounds,output_size)
    elina_abstract0_free(man,element)
    elina_manager_free(man)  

    return predicted_label, verified_flag, label_dict


def method_1 (nn, LB_N0, UB_N0, label, precompute, end_layer):

    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer 
    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(num_pixels)
    for i in range(num_pixels):
        elina_interval_set_double(itv[i],LB_N0[i],UB_N0[i])

    ## construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_pixels, itv)
    elina_interval_array_free(itv,num_pixels)


    m = Model("Solver")
    m.setParam( 'OutputFlag', False )

    linear_flag = False
    initialize_flag = True
    input_list = []
    output_list = []
    bounds_list = []
    linear_size = 0
    input_size = 0

    for layerno in range(numlayer):
        linear_flag = False

        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):

            if (nn.ffn_counter >= 0 and nn.ffn_counter <= end_layer): # perform normal interval computation
                linear_flag = True
            
            weights = nn.weights[nn.ffn_counter]
            biases = nn.biases[nn.ffn_counter]
            dims = elina_abstract0_dimension(man,element)
            num_in_pixels = weights.shape[1]
            num_out_pixels = weights.shape[0]

            bound_pair = []

            if (linear_flag and initialize_flag): # initialize input var
                bounds = elina_abstract0_to_box(man,element)

                for i in range(num_in_pixels):
                    inf = bounds[i].contents.inf.contents.val.dbl  # lower
                    sup = bounds[i].contents.sup.contents.val.dbl  # upper
                    #m.addConstr(input_var[i] >= inf)
                    #m.addConstr(input_var[i] <= sup)
                    bound_pair.append([inf, sup])
                    input_size += 1

                lb = [a[0] for a in bound_pair]
                ub = [a[1] for a in bound_pair]

                input_var = m.addVars(num_in_pixels, lb = lb, ub = ub, vtype=GRB.CONTINUOUS, name="input_" + str(nn.ffn_counter))


                input_list.append(input_var)
                bounds_list.append(bound_pair)
                initialize_flag = False

            logging.debug("Process Layer %d: in= %d, out= %d" % (layerno, num_in_pixels, num_out_pixels))

            dimadd = elina_dimchange_alloc(0,num_out_pixels)    
            for i in range(num_out_pixels):
                dimadd.contents.dim[i] = num_in_pixels
            elina_abstract0_add_dimensions(man, True, element, dimadd, False)
            elina_dimchange_free(dimadd)
            np.ascontiguousarray(weights, dtype=np.double)
            np.ascontiguousarray(biases, dtype=np.double)
            var = num_in_pixels
            # handle affine layer

            for i in range(num_out_pixels):
                tdim= ElinaDim(var)
                linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
                element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
                var+=1

            dimrem = elina_dimchange_alloc(0,num_in_pixels)
            for i in range(num_in_pixels):
                dimrem.contents.dim[i] = i
            elina_abstract0_remove_dimensions(man, True, element, dimrem)
            elina_dimchange_free(dimrem)
            
            # handle ReLU layer 

            if (linear_flag):
                lb = []
                ub = []
                rub = []

                bounds = elina_abstract0_to_box(man,element)

                for o in range(num_out_pixels):
                    inf = bounds[o].contents.inf.contents.val.dbl  # lower
                    sup = bounds[o].contents.sup.contents.val.dbl  # upper
                    lb.append(inf)
                    ub.append(sup)
                    if(sup <= 0):
                        rub.append(0)
                    else:
                        rub.append(sup)

                var_in = m.addVars(num_out_pixels, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="tmp_"+str(nn.ffn_counter))
                var = m.addVars(num_out_pixels, lb=0, ub=rub, vtype=GRB.CONTINUOUS, name="relu_"+str(nn.ffn_counter))

                bound_pairs = []
                output_list.append(var)

                for o in range (num_out_pixels):

                    coeff = defaultdict(float)

                    for i in range(num_in_pixels):
                        coeff[i] = weights[o][i]
                    
                    h = input_list[linear_size].prod(coeff) + biases[o]
                    m.addConstr(h == var_in[o])


                    m.setObjective(var_in[o], GRB.MINIMIZE)
                    m.optimize()
                    lower = m.objVal

                    m.setObjective(var_in[o], GRB.MAXIMIZE)
                    m.optimize()
                    upper = m.objVal

                    '''
                    if(linear_size >= 0):
                        for i in range(input_size):
                            v = m.getVars()
                            u = bounds_list[0][i][1]
                            l = bounds_list[0][i][0]
                            if (v[i].x != u and v[i].x != l):
                                print('%s %g != %g and %g' % (v[i].varName, v[i].x, u, l))
                    '''
                    bound_pairs.append([lower, upper])

                    #create an array of two linear constraints
                    lincons0_array = elina_lincons0_array_make(2)
                    
                    #Create a greater than or equal to inequality for the lower bound
                    lincons0_array.p[0].constyp = c_uint(ElinaConstyp.ELINA_CONS_SUPEQ)
                    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, 1)
                    cst = pointer(linexpr0.contents.cst)
                    
                    #plug the lower bound “a” here
                    elina_scalar_set_double(cst.contents.val.scalar, -lower)
                    linterm = pointer(linexpr0.contents.p.linterm[0])

                    #plug the dimension “i” here
                    linterm.contents.dim = ElinaDim(o)
                    coeff = pointer(linterm.contents.coeff)
                    elina_scalar_set_double(coeff.contents.val.scalar, 1)
                    lincons0_array.p[0].linexpr0 = linexpr0

                    #create a greater than or equal to inequality for the upper bound
                    lincons0_array.p[1].constyp = c_uint(ElinaConstyp.ELINA_CONS_SUPEQ)
                    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, 1)
                    cst = pointer(linexpr0.contents.cst)

                    #plug the upper bound “b” here
                    elina_scalar_set_double(cst.contents.val.scalar, upper)
                    linterm = pointer(linexpr0.contents.p.linterm[0])

                    #plug the dimension “i” here
                    linterm.contents.dim = ElinaDim(o)
                    coeff = pointer(linterm.contents.coeff)
                    elina_scalar_set_double(coeff.contents.val.scalar, -1)
                    lincons0_array.p[1].linexpr0 = linexpr0
                    #perform the intersection
                    element = elina_abstract0_meet_lincons_array(man,True,element,lincons0_array)                    

                for o in range (num_out_pixels):
                    if(nn.layertypes[layerno]=='ReLU'):

                        upper = bound_pairs[o][1]
                        lower = bound_pairs[o][0]

                        if (upper <= 0):
                            m.addConstr(output_list[linear_size][o] == 0)

                        elif (lower >= 0):
                            m.addConstr(output_list[linear_size][o] ,GRB.EQUAL, var_in[o])

                        else:
                            lambda_ = upper / (upper - lower)
                            mu_ = - lambda_ * lower

                            # method 1
                            m.addConstr( output_list[linear_size][o] >= 0 )
                            m.addConstr( output_list[linear_size][o] >= var_in  [o])            
                            m.addConstr( output_list[linear_size][o] <=     lambda_* var_in[o] + mu_ )
                    else:
                        m.addConstr(output_list[linear_size][o], GRB.EQUAL, var_in[o])

                input_list.append(var)
                logging.debug("==========Done Layer %d================" % layerno)

                linear_size +=1
            
            if(nn.layertypes[layerno]=='ReLU'):
                element = relu_box_layerwise(man,True,element,0, num_out_pixels)
                #elina_abstract0_fprint(cstdout, man, element, None)


            nn.ffn_counter+=1 

        else:
           print(' net type not supported')

    output_size = num_out_pixels

    # if epsilon is zero, try to classify else verify robustness 
    
    verified_flag = True
    predicted_label = 0

    for j, val in precompute:
        if(j!=label):
            m.setObjective(input_list[linear_size][label] - 
                           input_list[linear_size][j], GRB.MINIMIZE)
            m.optimize()
            #m.write('model_'+str(j)+'.lp')

            status = m.Status
            if status == GRB.Status.INF_OR_UNBD or \
               status == GRB.Status.INFEASIBLE or \
               status == GRB.Status.UNBOUNDED :
                if (status == GRB.Status.INF_OR_UNBD): print('INF_OR_UNBD')
                elif  (status == GRB.Status.INFEASIBLE): print('INFEASIBLE')
                elif (status == GRB.Status.UNBOUNDED): print('UNBOUNDED')

                break

            else:
                obj = m.objVal
                print('Obj: %d %g' % (j, obj))

            if(m.objVal <= 0):
                #for v in m.getVars():
                #    print('%s %g' % (v.varName, v.x))

                predicted_label = label
                verified_flag = False
                break

    elina_interval_array_free(bounds,output_size)
    elina_abstract0_free(man,element)
    elina_manager_free(man)        
    return predicted_label, verified_flag
    
def method_2 (nn, LB_N0, UB_N0, label, precompute, end_layer):

    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer 
    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(num_pixels)
    for i in range(num_pixels):
        elina_interval_set_double(itv[i],LB_N0[i],UB_N0[i])

    ## construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_pixels, itv)
    elina_interval_array_free(itv,num_pixels)


    m = Model("Solver")
    m.setParam( 'OutputFlag', False )

    linear_flag = False
    initialize_flag = True
    input_list = []
    bounds_list = []
    linear_size = 0
    input_index = []
    output_scale_U = []
    output_scale_L = []
    final_output_U = []
    final_output_L = []

    layer_affine_talbe = []
    var_cnt = 1


    for layerno in range(numlayer):
        linear_flag = False
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):

            if (nn.ffn_counter >= 0 and nn.ffn_counter <= end_layer): # perform normal interval computation
                linear_flag = True
            

            weights = nn.weights[nn.ffn_counter]
            biases = nn.biases[nn.ffn_counter]
            dims = elina_abstract0_dimension(man,element)
            num_in_pixels = weights.shape[1]
            num_out_pixels = weights.shape[0]

            bound_pair = []

            if (linear_flag and initialize_flag): # initialize input var
                bounds = elina_abstract0_to_box(man,element)
                input_var = m.addVars(num_in_pixels, vtype=GRB.CONTINUOUS, name="input_" + str(nn.ffn_counter))
                indice = []

                for i in range(num_in_pixels):
                    inf = bounds[i].contents.inf.contents.val.dbl  # lower
                    sup = bounds[i].contents.sup.contents.val.dbl  # upper
                    m.addConstr(input_var[i] >= inf)
                    m.addConstr(input_var[i] <= sup)
                    bound_pair.append([inf, sup])

                    t = defaultdict(float, {0: 0, var_cnt: 1})
                    indice.append(t)
                    var_cnt += 1

                input_list.append(input_var)
                bounds_list.append(bound_pair)
                input_index.append(indice)
                initialize_flag = False

            logging.debug("Process Layer %d: in= %d, out= %d" % (layerno, num_in_pixels, num_out_pixels))

            dimadd = elina_dimchange_alloc(0,num_out_pixels)    
            for i in range(num_out_pixels):
                dimadd.contents.dim[i] = num_in_pixels
            elina_abstract0_add_dimensions(man, True, element, dimadd, False)
            elina_dimchange_free(dimadd)
            np.ascontiguousarray(weights, dtype=np.double)
            np.ascontiguousarray(biases, dtype=np.double)
            var = num_in_pixels
            # handle affine layer

            for i in range(num_out_pixels):
                tdim= ElinaDim(var)
                linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
                element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
                var+=1

            dimrem = elina_dimchange_alloc(0,num_in_pixels)
            for i in range(num_in_pixels):
                dimrem.contents.dim[i] = i
            elina_abstract0_remove_dimensions(man, True, element, dimrem)
            elina_dimchange_free(dimrem)
            
            # handle ReLU layer 


            if (linear_flag):

                bound_pairs = []
                indice = []
                relu_scale_L = []
                relu_scale_U = []
                affine_table = []

                for o in range (num_out_pixels):

                    t1 = defaultdict(float) # for upper
                    t2 = defaultdict(float) # for lower

                    u1 = defaultdict(float) # for upper
                    u2 = defaultdict(float) # for lower

                    if (linear_size == 0):
              
                        layer_weight= weights
                        layer_size = num_in_pixels
                        
                        for j in range(layer_size):
                            c =  layer_weight[o][j]
                            f = input_index[linear_size][j]
                            for key in f:
                                t1[key] += c * f[key]
                                t2[key] += c * f[key]
                        t1[0] += biases[o]
                        t2[0] += biases[o]
                        u1 = t1
                        u2 = t2
                    else:

                        for i in reversed(range(linear_size)):

                            if (i == linear_size - 1):

                                layer_weight= nn.weights[nn.ffn_counter]
                                layer_size = layer_weight.shape[1]
                        
                                for j in range(layer_size):
                                    c =  layer_weight[o][j]
                                    f = input_index[i+1][j]
                                    for key in f:
                                        t1[key] += c * f[key]
                                        t2[key] += c * f[key]
                                t1[0] += biases[o]
                                t2[0] += biases[o]
                                u1 = t1 
                                u2 = t2

                            c1 = list(t1.values())
                            c2 = list(t2.values())

                            v1 = defaultdict(float) # for upper
                            v2 = defaultdict(float) # for lower

                            for j in range(len(c1)):
                                if (not j==0) :
                                    if (c1[j]>=0):
                                        f1 = output_scale_U[i][j-1]
                                    else:
                                        f1 = output_scale_L[i][j-1]
                                    for key in f1:
                                        v1 [key] += f1[key] * c1[j]

                                    if (c2[j] >=0):
                                        f2 = output_scale_L[i][j-1]
                                    else:
                                        f2 = output_scale_U[i][j-1]
                                    for key in f2:
                                        v2 [key] += f2[key] * c2[j]
                            v1[0] += c1[0]
                            v2[0] += c2[0]
                            t1 = v1
                            t2 = v2



                    upper = t1[0]
                    lower = t2[0]
                    for i in t1:
                        if i != 0:
                            if t1[i] >=0:
                                upper += bounds_list[0][i-1][1] * t1[i]
                            else:
                                upper += bounds_list[0][i-1][0] * t1[i]
                    for i in t2:
                        if i != 0:
                            if t2[i] >=0:
                                lower += bounds_list[0][i-1][0] * t2[i]
                            else:
                                lower += bounds_list[0][i-1][1] * t2[i]

                    #print("#%d [lower =%.10f, upper=%.10f]"%(o, lower, upper))
                    bound_pair.append([lower, upper])

                    
                    #create an array of two linear constraints
                    lincons0_array = elina_lincons0_array_make(2)
                    
                    #Create a greater than or equal to inequality for the lower bound
                    lincons0_array.p[0].constyp = c_uint(ElinaConstyp.ELINA_CONS_SUPEQ)
                    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, 1)
                    cst = pointer(linexpr0.contents.cst)
                    
                    #plug the lower bound “a” here
                    elina_scalar_set_double(cst.contents.val.scalar, -lower)
                    linterm = pointer(linexpr0.contents.p.linterm[0])

                    #plug the dimension “i” here
                    linterm.contents.dim = ElinaDim(o)
                    coeff = pointer(linterm.contents.coeff)
                    elina_scalar_set_double(coeff.contents.val.scalar, 1)
                    lincons0_array.p[0].linexpr0 = linexpr0

                    #create a greater than or equal to inequality for the upper bound
                    lincons0_array.p[1].constyp = c_uint(ElinaConstyp.ELINA_CONS_SUPEQ)
                    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, 1)
                    cst = pointer(linexpr0.contents.cst)

                    #plug the upper bound “b” here
                    elina_scalar_set_double(cst.contents.val.scalar, upper)
                    linterm = pointer(linexpr0.contents.p.linterm[0])

                    #plug the dimension “i” here
                    linterm.contents.dim = ElinaDim(o)
                    coeff = pointer(linterm.contents.coeff)
                    elina_scalar_set_double(coeff.contents.val.scalar, -1)
                    lincons0_array.p[1].linexpr0 = linexpr0
                    #perform the intersection
                    element = elina_abstract0_meet_lincons_array(man,True,element,lincons0_array)                    
                    
                  
                    indice.append(defaultdict(float, {0: 0, var_cnt: 1}))
                    var_cnt += 1

                    if(nn.layertypes[layerno]=='ReLU'):
                        if (upper <= 0):
                            relu_scale_L.append(defaultdict(float, {0: 0}))                       
                            relu_scale_U.append(defaultdict(float, {0: 0}))
    
                        elif (lower >= 0):
                            relu_scale_L.append(u2)                       
                            relu_scale_U.append(u1)
    
                        else:
                            lambda_ = upper / (upper - lower)
                            mu_ = - lambda_ * lower
                            w = defaultdict(float)   
                            for i in u1:
                                w[i] = lambda_ * u1[i]
                            w[0] += mu_
                            if(upper <= - lower):
                                relu_scale_L.append(defaultdict(float, {0: 0}))                       
                            else:
                                relu_scale_L.append(u2)                       
    
                            relu_scale_U.append(w)
                    else:
                        relu_scale_L.append(u2)                       
                        relu_scale_U.append(u1)


                input_index.append(indice) 
                output_scale_L.append(relu_scale_L)
                output_scale_U.append(relu_scale_U)
                logging.debug("==========Done Layer %d================" % layerno)
                linear_size +=1
            
            if(nn.layertypes[layerno]=='ReLU'):
                element = relu_box_layerwise(man,True,element,0, num_out_pixels)
                #elina_abstract0_fprint(cstdout, man, element, None)

            nn.ffn_counter+=1 

        else:
           print(' net type not supported')

    output_size = num_out_pixels

    # if epsilon is zero, try to classify else verify robustness 
    
    verified_flag = True
    predicted_label = 0

    for j, val in precompute:
        if(j!=label):
            coeff = defaultdict(float)
            t = defaultdict(float)
            f1 = output_scale_L[linear_size-1][label]
            f2 = output_scale_L[linear_size-1][j]
            for key in f1:
                t [key] += f1[key]
            for key in f2:
                t [key] -= f2[key]

            #print(t.items())
            for i in reversed(range(linear_size-1)):
                
                c = list(t.values())
                v = defaultdict(float) # for upper

                for jj in range(len(c)):
                    if (not jj==0) :
                        if (c[jj] >= 0):
                            f = output_scale_L[i][jj-1]
                        else:
                            f = output_scale_U[i][jj-1]
                        for key in f:
                            v [key] += f[key] * c[jj]
                v[0] += c[0]

                t = v

            for i in t:
                if (i!=0): coeff[i-1] = t[i]
            
            obj_funct = input_list[0].prod(coeff) + t[0]

            m.setObjective(obj_funct, GRB.MINIMIZE)
            m.optimize()

            status = m.Status
            if status == GRB.Status.INF_OR_UNBD or \
               status == GRB.Status.INFEASIBLE or \
               status == GRB.Status.UNBOUNDED :
                if (status == GRB.Status.INF_OR_UNBD): print('INF_OR_UNBD')
                elif  (status == GRB.Status.INFEASIBLE): print('INFEASIBLE')
                elif (status == GRB.Status.UNBOUNDED): print('UNBOUNDED')

                break

            else:
                obj = m.objVal
                print('Obj: %d %g' % (j, obj))

            if(m.objVal <= 0):

                predicted_label = label
                verified_flag = False
                break

    elina_interval_array_free(bounds,output_size)
    elina_abstract0_free(man,element)
    elina_manager_free(man)        
    return predicted_label, verified_flag
        
def method_3 (nn, LB_N0, UB_N0, label, precompute, end_layer):

    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer 
    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(num_pixels)
    for i in range(num_pixels):
        elina_interval_set_double(itv[i],LB_N0[i],UB_N0[i])

    ## construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_pixels, itv)
    elina_interval_array_free(itv,num_pixels)

    linear_flag = False
    initialize_flag = True
    stop = False
    variable_list = []
    output_size = 0
    epsilon_cnt = 1

    m = Model("Solver")
    m.setParam( 'OutputFlag', False )

    for layerno in range(numlayer):
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):

            if (nn.ffn_counter >= 0): # perform normal interval computation
                linear_flag = True

            if (nn.ffn_counter > end_layer):
                stop = True

            weights = nn.weights[nn.ffn_counter]
            biases = nn.biases[nn.ffn_counter]
            dims = elina_abstract0_dimension(man,element)
            num_in_pixels = weights.shape[1]
            num_out_pixels = weights.shape[0]

            if (linear_flag and initialize_flag): # initialize input var
                bounds = elina_abstract0_to_box(man,element)
                for i in range(num_in_pixels):
                    inf = bounds[i].contents.inf.contents.val.dbl  # lower
                    sup = bounds[i].contents.sup.contents.val.dbl  # upper

                    if (sup <= 0):
                        mid = 0
                        t = defaultdict(float, {0: mid})
                        variable_list.append(t)
                    elif (inf >= 0):
                        mid = (sup + inf) / 2
                        eps = sup - mid
                        t = defaultdict(float, {0: mid, epsilon_cnt: eps})
                        variable_list.append(t)
                        epsilon_cnt += 1

                    else:
                        mid = (sup + inf) / 2
                        eps1 = sup - mid
                        lambda_ = sup / (sup - inf)
                        mu_ = - lambda_ * inf / 2.0
                        mid  += mu_
                        eps1 *= lambda_
                        eps2  = mu_
                        t = defaultdict(float, {0: mid, epsilon_cnt: eps1, epsilon_cnt+1: eps2})
                        variable_list.append(t)
                        epsilon_cnt += 2
                initialize_flag = False

            logging.debug("Process Layer %d: in= %d, out= %d" % (layerno, num_in_pixels, num_out_pixels))

            dimadd = elina_dimchange_alloc(0,num_out_pixels)    
            for i in range(num_out_pixels):
                dimadd.contents.dim[i] = num_in_pixels
            elina_abstract0_add_dimensions(man, True, element, dimadd, False)
            elina_dimchange_free(dimadd)
            np.ascontiguousarray(weights, dtype=np.double)
            np.ascontiguousarray(biases, dtype=np.double)
            var = num_in_pixels
            # handle affine layer

            for i in range(num_out_pixels):
                tdim= ElinaDim(var)
                linexpr0 = generate_linexpr0(weights[i],biases[i],num_in_pixels)
                #elina_linexpr0_print(linexpr0, None)
                element = elina_abstract0_assign_linexpr_array(man, True, element, tdim, linexpr0, 1, None)
                var+=1
                
            dimrem = elina_dimchange_alloc(0,num_in_pixels)
            for i in range(num_in_pixels):
                dimrem.contents.dim[i] = i
            elina_abstract0_remove_dimensions(man, True, element, dimrem)
            elina_dimchange_free(dimrem)
            
            # handle ReLU layer 
            if(nn.layertypes[layerno]=='ReLU'):
                #elina_abstract0_fprint(cstdout, man, element, None)
                element = relu_box_layerwise(man,True,element,0, num_out_pixels)
            

            if (linear_flag):
                # compute Affine
                output_variable = []
                for o in range (num_out_pixels):
                    t = defaultdict(float)
                    for c, f in zip (weights[o], variable_list):
                        for i in f:
                            t[i] += c * f[i]
                    t[0] += biases[o]
                    output_variable.append(t)

                relu_variable = []
                for f in output_variable:
                    if(nn.layertypes[layerno]=='ReLU'):
                        a = sum(abs(f[i]) for i in f if i != 0)
                        u = f[0] + a
                        l = f[0] - a
                        if u <= 0:
                            t = defaultdict(float, {0: 0})
                            relu_variable.append(t)
                        elif l >= 0:
                            relu_variable.append(f)
                        else:
                            lambda_ = u / (u - l)
                            mu_ = - lambda_ * l / 2.0
                            t = defaultdict(float)   
                            for i in f:
                                t[i] = lambda_ * f[i]
                            t [epsilon_cnt] = mu_
                            t [0] += mu_
                            epsilon_cnt += 1
                            relu_variable.append(t)
                    else:
                        relu_variable.append(f)

                variable_list = relu_variable 

            nn.ffn_counter+=1
            logging.debug("==========Done Layer %d================" % layerno)


        else:
           print(' net type not supported')

        output_size = num_out_pixels
           
    # if epsilon is zero, try to classify else verify robustness 
    
    verified_flag = True
    predicted_label = 0
    gurobi_var = m.addVars(epsilon_cnt-1, lb=-1, ub=1, vtype=GRB.CONTINUOUS, name="input")

    for j, val in precompute:
        if(j!=label):
            t = defaultdict(float)
            coeff = defaultdict(float)

            f1 = variable_list [label]
            f2 = variable_list [j]

            for i in f1:
                t[i] += f1[i]
            for i in f2:
                t[i] -= f2[i]

            for i in t:
                if (i!=0): coeff[i-1] = t[i]
            
            obj_funct = gurobi_var.prod(coeff) + t[0]
                
            m.setObjective(obj_funct, GRB.MINIMIZE)
            m.optimize()
            
            status = m.Status
            if status == GRB.Status.INF_OR_UNBD or \
               status == GRB.Status.INFEASIBLE or \
               status == GRB.Status.UNBOUNDED :
                if (status == GRB.Status.INF_OR_UNBD): logging.debug('INF_OR_UNBD')
                elif  (status == GRB.Status.INFEASIBLE): logging.debug('INFEASIBLE')
                elif (status == GRB.Status.UNBOUNDED): logging.debug('UNBOUNDED')
                verified_flag = False
                break

            else:
                obj = m.objVal
                logging.debug('Obj: %d %g' % (j, obj))

            if(obj < 0):
                predicted_label = label
                verified_flag = False
                break

    elina_interval_array_free(bounds,output_size)
    elina_abstract0_free(man,element)
    elina_manager_free(man)        
    return predicted_label, verified_flag  

def analyer_mananger ( nn, LB_N0, UB_N0, label, epsilon, precompute):

    n_layers = nn.numlayer

    n_parameters = 0
    
    n_compute = 0

    for l in range(1, n_layers):
        n_prev = nn.weights[l-1].shape[1]
        n_in = nn.weights[l].shape[1]
        n_out = nn.weights[l].shape[0]
        n_compute +=  n_prev * n_in * n_out * (n_layers-l)
        n_parameters += n_in + n_out

    logging.debug('# layer: %d' % n_layers)
    logging.debug('# parameters: %d' % n_parameters)
    logging.debug('# bottleneck: %e' % n_compute)
    logging.debug('epsilon: %g' % epsilon)

    # Our heuristic
    
    if (n_parameters < 2000): # most precise but slow
        #logging.debug('prefrom heuristic 1\n')
        predicted_label, verified_flag = method_1 (nn, LB_N0, UB_N0, label, precompute, n_layers-1)
    elif (n_compute < 1e9): # fast but lose precision
        #logging.debug('prefrom heuristic 2\n')
        predicted_label, verified_flag = method_2 (nn, LB_N0, UB_N0, label, precompute, n_layers-1)
    else: # faster but lose more precisison
        #logging.debug('prefrom heuristic 3\n')
        predicted_label, verified_flag = method_3 (nn, LB_N0, UB_N0, label, precompute, n_layers-2)
    

    return predicted_label, verified_flag





def analyze(nn, LB_N0, UB_N0, label, epsilon, precompute=None):

    if(LB_N0[0]==UB_N0[0]):
        predicted_label, verified_flag, label_dict = prerun (nn, LB_N0, UB_N0, label)

    else:
        predicted_label, verified_flag = analyer_mananger (nn, LB_N0, UB_N0, label, epsilon, precompute)
        label_dict = []


    return predicted_label, verified_flag, label_dict



if __name__ == '__main__':
    from sys import argv
    if len(argv) < 3 or len(argv) > 4:
        print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
        exit(1)

    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])
    #c_label = int(argv[4])
    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    with open(specname, 'r') as specfile:
        specstring = specfile.read()
    nn = parse_net(netstring)
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low,0)

    logging.debug('Pre-predict image label...')

    label, _, label_dict = analyze(nn,LB_N0,UB_N0,0, epsilon)
    
    logging.debug(label_dict)
    print('\n')

    start = time.time()
    if(label==int(x0_low[0])):
        LB_N0, UB_N0 = get_perturbed_image(x0_low,epsilon)
        _, verified_flag, _ = analyze(nn,LB_N0,UB_N0,label, epsilon, label_dict)
        if(verified_flag):
            print("verified")
        else:
            print("can not be verified")  
    else:
        print("image not correctly classified by the network. expected label ",int(x0_low[0]), " classified label: ", label)
    end = time.time()
    print("analysis time: ", (end-start), " seconds")
    

