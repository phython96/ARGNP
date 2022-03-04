import os
from graphviz import Digraph
from collections import namedtuple
from hydra.utils import get_original_cwd, to_absolute_path
Genotype = namedtuple('Genotype', 'V E')

def plot_genotype(genotype):
    g = Digraph("G", format="pdf")
    g.attr(rankdir='TD', size='8,5')
    # import ipdb; ipdb.set_trace()
    nums = 0
    for Vi, Vj, Ek, Op in genotype.V:
        if Vi > nums:
            nums = Vi

    g.node('V0', 'V_{in0}', shape="rect", style="filled", color="black", fillcolor="lightblue", penwidth = '2')
    g.node('V1', 'V_{in1}', shape="rect", style="filled", color="black", fillcolor="lightblue", penwidth = '2')
    g.node('Vout', 'Vout', shape="rect", style="filled", color="black", fillcolor="lavender", penwidth = '2')
    for i in range(2, nums + 1):
        g.node(f'V{i}', f'V{i-1}', shape="rect", style="filled", color="black", fillcolor="lightcyan", penwidth = '2')
        g.edge(f'V{i}', 'Vout', arrowhead="empty")

    g.node('E0', 'E_{in0}', shape="rect", style="filled", color="black", fillcolor="lightcoral", penwidth = '2')
    g.node('E1', 'E_{in1}', shape="rect", style="filled", color="black", fillcolor="lightcoral", penwidth = '2')
    g.node('Eout', 'Eout', shape="rect", style="filled", color="black", fillcolor="peachpuff", penwidth = '2')
    for i in range(2, nums + 1):
        g.node(f'E{i}', f'E{i-1}', shape="rect", style="filled", color="black", fillcolor="mistyrose", penwidth = '2')    
        g.edge(f'E{i}', f'Eout', arrowhead="empty")  
    
    for Vi, Vj, Ek, Op in genotype.V:
        if Ek == 0:
            Ek = "_{in0}"
        elif Ek == 1:
            Ek = "_{in1}"
        else:
            Ek -= 1
        ext = f'(+E{Ek})' if Op not in ['V_I', 'E_I'] else ''
        # color = 'red' if flag else 'black'
        # weight = '3' if flag else '1'
        g.edge(f'V{Vj}', f'V{Vi}', f'{Op}{ext}', arrowhead="empty")
    
    for Ei, Vj, Ek, Op in genotype.E:
        if Vj == 0:
            Vj = "_{in0}"
        elif Vj == 1:
            Vj = "_{in1}"
        else:
            Vj -= 1
        ext = f'(+V{Vj})' if Op not in ['V_I', 'E_I'] else ''
        # color = 'red' if flag else 'black'
        # weight = '3' if flag else '1'
        g.edge(f'E{Ek}', f'E{Ei}', f'{Op}{ext}', arrowhead="empty")

    return g


def plot_genotypes(args, i_epoch, geno): 
    root = args.ds.arch_save
    for i in range(args.basic.nb_layers): 
        cdir = os.path.join(root, args.ds.data, str(i_epoch), f"cell_{i}")
        g = plot_genotype(geno[i])
        g.render(format = 'pdf', filename = cdir)
        cdir = os.path.join(get_original_cwd(), root, args.ds.data, str(i_epoch), f"cell_{i}")
        g.render(format = 'pdf', filename = cdir)
    
    path = os.path.join(root, args.ds.data, str(i_epoch), "cell_geno.txt")
    with open(path, 'w') as f:
        f.write(str(geno))

    path = os.path.join(get_original_cwd(), root, args.ds.data, str(i_epoch), "cell_geno.txt")
    with open(path, 'w') as f:
        f.write(str(geno))


def main():
    pname = 'modelnet10'
    save  = 'pics'
    # path  = f'archs2/start/repeat1/{pname}/25/cell_geno.txt'
    path  = f'archs2/start2/repeat3/{pname}/45/cell_geno.txt'
    with open(path) as f:
        genotypes = eval(f.read())
    g = plot_genotype(genotypes[0])
    g.render(format = 'pdf', filename = os.path.join(save, "tmp"))

if __name__ == '__main__':
    main()