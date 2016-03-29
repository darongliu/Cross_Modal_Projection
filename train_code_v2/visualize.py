import tsne
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def data_visualize(data,triplet,all_info) :
    print "Run tsne"
    Y = tsne.tsne(data, 2, 50, 20.0)

    all_entity   = all_info[0] 
    all_relation = all_info[1]
    idx2name     = all_info[2]
    name2idx     = all_info[3]

    for pos in range(3) :
        print 'draw pos',pos
        all_possible_label = set()
        for tri in triplet :
            all_possible_label.add(tri[pos])

        colors = cm.rainbow(np.linspace(0, 1, len(all_possible_label)))
        for label_idx ,c in zip(all_possible_label,colors) :
            temp  = Y
            count = 0
            for tri_idx in range(len(triplet)) :
                if triplet[tri_idx][pos] == label_idx :
                    count = count + 1
                else :
                    temp = np.delete(temp,count,0)
            plt.scatter(temp[:, 0], temp[:, 1], marker = 'o', color=c)
        plt.show()

#    for i, tri in enumerate(triplet):
#        name = id2name[tri[0]] + ' ' + id2name[tri[1]] + ' ' + id2name[tri[2]]
#        plt.annotate(name, (Y[i,0],Y[i,1]))

#    plt.show()
