import numpy as np
import matplotlib.pyplot as plt
import math


class Individual(object):
    def __init__(self, dv_list: list, obj_list):
        # self.dv = np.array(dv_list)
        self.obj = np.array(obj_list)
        # self.n_dv = len(self.dv)
        self.n_obj = len(self.obj)


    def dominate(self, other) -> bool:
        # print('dominate..........................')
        if not isinstance(other, Individual):
            Exception("not indiv.")

        if all(s <= o for s, o in zip(self.obj, other.obj)) and \
                any(s != o for s, o in zip(self.obj, other.obj)):
            return True
        return False


def indiv_sort(population, key=-1):
    # print('indiv_sort......................')
    popsize = len(population)
    if popsize <= 1:
        return population
    # print('population', population)
    pivot = population[0]  # ！！！！！！！！
    left = []
    right = []
    for i in range(1, popsize):
        indiv = population[i]
        # print('indiv',indiv.obj, indiv.obj[key], pivot.obj[key])
        if indiv.obj[key] <= pivot.obj[key]:
            left.append(indiv)
        else:
            right.append(indiv)
    # print('left,right', left, right)
    left = indiv_sort(left, key)
    right = indiv_sort(right, key)

    center = [pivot]
    # print('center', center)
    return left + center + right


class NonDominatedSort(object):

    def __init__(self):
        pass
        # self.pop = pop

    def sort(self, population: list, return_rank=False):
        # print('NonDominatedSort..............................')
        popsize = len(population)

        is_dominated = np.empty((popsize, popsize), dtype=np.bool)
        num_dominated = np.zeros(popsize, dtype=np.int64)
        mask = np.empty(popsize, dtype=np.bool)
        rank = np.zeros(popsize, dtype=np.int64)

        count_true = 0
        count_false = 0
        for i in range(popsize):
            for j in range(popsize):
                # if i == j:
                #     continue

                dom = population[j].dominate(population[i])
                if dom == True:
                    count_true += 1
                else:
                    count_false += 1
                is_dominated[i, j] = (i != j) and dom


        is_dominated.sum(axis=(1,), out=num_dominated)
        # print('num_dominated', num_dominated)
        # print('is_dominated', is_dominated)

        fronts = []
        limit = popsize
        index_list=[]
        for r in range(popsize):
            front = []
            for i in range(popsize):
                is_rank_ditermined = not (rank[i] or num_dominated[i])
                mask[i] = is_rank_ditermined
                if is_rank_ditermined:
                    rank[i] = r + 1
                    front.append(population[i])
            index_list.append(i)
            fronts.append(front)

            limit -= len(front)
            # print('front', front)
            # print('limit', limit)

            if return_rank:
                if rank.all():
                    return rank
            elif limit <= 0:
                return fronts,index_list

            # print(np.sum(mask & is_dominated))
            num_dominated -= np.sum(mask & is_dominated, axis=(1,))

        raise Exception("Error: reached the end of function")


class HyperVolume(object):
    def __init__(self, pareto, ref_points: list):
        self.pareto = pareto
        self.pareto_sorted = indiv_sort(self.pareto)
        self.ref_point = np.ones(pareto[0].n_obj)
        self.ref_point = np.array(ref_points)

        self.obj_dim = pareto[0].n_obj
        self.volume = 0

        self.calcpoints = []

    def set_refpoint(self, opt="minimize"):
        pareto_arr = []
        for indiv in self.pareto:
            pareto_arr.append(indiv.obj)
        pareto_arr = np.array(pareto_arr)

        # print('pareto_arr', pareto_arr)
        minmax = max
        if opt == "maximize":
            minmax = min

        for i in range(len(self.ref_point)):
            self.ref_point[i] = minmax(pareto_arr[:, i])

    def hso(self):
        # print('hso............................')
        pl, s = self.obj_dim_sort(self.pareto)
        s = [pl, s]
        for k in range(self.obj_dim):
            s_dash = []

    def calculate(self, obj_dim):
        pass

    def calc_2d(self):
        if len(self.ref_point) != 2:
            return NotImplemented

        vol = 0
        b_indiv = None

        for i, indiv in enumerate(self.pareto_sorted):
            if i == 0:
                x = (self.ref_point[0] - indiv.obj[0])
                y = (self.ref_point[1] - indiv.obj[1])
            else:
                x = (b_indiv.obj[0] - indiv.obj[0])
                y = (self.ref_point[1] - indiv.obj[1])

            self.calcpoints.append([x, y])
            vol += x * y
            b_indiv = indiv
            # print(f"vol:{vol:.10f}  x:{x:.5f}  y:{y:.5f}")

        self.volume = vol
        self.calcpoints = np.array(self.calcpoints)
        return vol

    def obj_dim_sort(self, dim=-1):
        pareto_arr = []
        for indiv in self.pareto:
            pareto_arr.append(indiv.obj)

        pareto_arr = np.array(pareto_arr)
        res_arr = pareto_arr[pareto_arr[:, dim].argsort(), :]
        self.pareto_sorted = res_arr

        # print('pareto_arr,res_arr', pareto_arr, )

        return res_arr, res_arr[:, dim]


def indiv_plot(population: list, color=None):
    evals = []
    for indiv in (population):
        # print(indiv)
        evals.append(indiv.obj)

    evals = np.array(evals)
    print('evals[:, 0]:',evals[:, 0])
    print('evals[:, 1]:',evals[:, 1])
    plt.scatter(evals[:, 0], evals[:, 1], c=color)


def data_save(pareto, vol, ref_point, fname, ext="txt"):
    pareto_arr = []
    for indiv in pareto:
        pareto_arr.append(indiv.obj)
    pareto_arr = np.array(pareto_arr)

    delimiter = " "
    if ext == "csv":
        delimiter = ","

    np.savetxt(fname + "_pareto." + ext, pareto_arr, delimiter=delimiter)
    with open(fname + "_HV." + ext, "w") as f:
        f.write("#HyperVolume\n")
        f.write(f"{vol}\n")

        f.write("#ref_point\n")
        for p in ref_point:
            f.write(f"{p}, ")
        f.write("\n")


def ref_point(input_fname):
    x_list = []
    y_list = []
    with open(input_fname, 'r') as f:
        for line in f:
            # print(line.strip().split())
            x, y = (line.strip().split(','))
            x_list.append(float(x))
            y_list.append(float(y))
    # print(x_list, '\n', y_list, '\n', len(x_list))
    x = np.array(x_list)
    y = np.array(y_list)
    x = x.astype(float)
    y = y.astype(float)
    # print(x,'\n',y)
    return [math.ceil(np.max(x)), math.ceil(np.max(y))]
    # print('x:max:', np.max(x))
    # print('x:min:', np.min(x))
    # print('y:max:', np.max(y))
    # print('y:min:', np.min(y))


def main():
    # print('main....................')
    # input_fname = "/tmp/pycharm_project_101/src/test_table.txt"  # input file name
    input_fname='/tmp/pycharm_project_77/predict_original_all_other2.txt'

    output_fname = "test_result_data"  # result file name
    ext = "txt"  # output
    # ref_points = ref_point(input_fname)
    ref_points=[3,1]

    # non-dominated-sort
    # skiprows=1
    ls = open(input_fname).readlines()
    newTxt = ""
    all_list=[]
    for line in ls:
        a,b=line.strip().split(',')
        newTxt = newTxt + " ".join(line.split(','))
        all_list.append([float(a),float(b)])
    # print(newTxt)
    fo = open(input_fname+'_copy', 'w')
    fo.write(newTxt)
    fo.close()

    dat = np.loadtxt(input_fname+'_copy')
    sortfunc = NonDominatedSort()
    population = []
    # [1:6],[6:]
    for s in dat:
        population.append(Individual([], s))
        # print(population[-1].__dict__)

    front,index_list = sortfunc.sort(population)
    print('main results', front)

    pareto_first = front[0]
    pareto_last=front[-1]
    num_font = 0
    for i in range(len(front)):
        num_font += len(front[0])
    # print('num_front', num_font)

    print("Number of pareto solutions: ", len(pareto_first))
    # print(index_list)
    # print('front',front[0][0].obj)
    # print(front[0][0],front[0][0][0])
    # print(all_list)
    front_index_list=[]
    for i in range(len(front[0])):
        front_index_list.append(all_list.index([front[0][i].obj[0],front[0][i].obj[1]]))
    print('front_index_list',front_index_list)

    front_index_list1 = []
    for i in range(len(front[1])):
        front_index_list1.append(all_list.index([front[1][i].obj[0], front[1][i].obj[1]]))
    print('front_index_list1', front_index_list1)
    # calculate HV
    hypervol_best = HyperVolume(pareto_first, ref_points)
    hypervol_wrost=HyperVolume(pareto_last, ref_points)
    vol_first = hypervol_best.calc_2d()
    vol_last=hypervol_wrost.calc_2d()
    print("ref_point: ", hypervol_best.ref_point)
    print("HV: ", vol_first)
    data_save(hypervol_best.pareto_sorted, vol_first, hypervol_best.ref_point, output_fname, ext=ext)

    # plot all indiv(blue) and pareto indiv(red)
    indiv_plot(population, color='y')

    indiv_plot(front[0], color="Red")
    indiv_plot(front[1], color="green")
    plt.xlim(-1.5,3)
    plt.ylim(0,1)
    plt.xlabel('D1_R(mic)')

    plt.ylabel('D2_R(hemo)')
    # plt.scatter(hypervol.calcpoints[:,0], hypervol.calcpoints[:,1], "*")
    print("HV: ",vol_first)
    print(hypervol_best.calcpoints)
    plt.title(str(vol_first))
    plt.savefig('/tmp/pycharm_project_77/'+'pareto'+'.png', dpi=200, bbox_inches='tight', transparent=False)
    plt.show()
    return vol_first,vol_last



if __name__ == "__main__":
    main()


