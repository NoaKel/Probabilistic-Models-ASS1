import sys
import math
from collections import Counter

def main():
    development_set_filename = sys.argv[1]
    test_set_filename = sys.argv[2]
    input_word = sys.argv[3]
    output_file_name = sys.argv[4]

    output = []
    output.append("Noa Yehezkel Lubin 305097552 Yehudith *****")

    ### Question 1 ###
    # Output 1
    output.append("Output1:"+"\t"+development_set_filename)
    # Output 2
    output.append("Output2:"+"\t"+test_set_filename)
    # Output 3
    output.append("Output3:"+"\t"+input_word)
    # Output 4
    output.append("Output4:"+"\t"+output_file_name)
    # Output 5
    V = 300000
    output.append("Output5:"+"\t"+str(V))
    # Output 6
    output.append("Output6:"+"\t"+str(pUniform(V)))

    ### Question 2 ###
    # Output 7
    words = parseDevelop(development_set_filename)
    output.append("Output7:"+"\t"+str(len(words)))

    ### Question 3 ###
    # Output 8 & 9
    idx = int(round(len(words) * 0.9))
    Lidstone_training, Lidstone_validation = words[:idx], words[idx:]
    S = len(Lidstone_training)
    output.append("Output8:"+"\t"+str(len(Lidstone_validation)))
    output.append("Output9:"+"\t"+str(S))
    # Output 10
    Lidstone_training_counter = Counter(Lidstone_training)
    output.append("Output10:"+"\t"+str(len(Lidstone_training_counter)))
    # Output 11
    C_x = Lidstone_training_counter[input_word]
    C_unk = Lidstone_training_counter["unseen-word"]
    output.append("Output11:"+"\t"+str(C_x))
    # Output 12
    output.append("Output12:"+"\t"+str(pLidstone(V,S,C_x)))
    # Output 13
    output.append("Output13:"+"\t"+str(pLidstone(V,S,C_unk)))
    # Output 14
    output.append("Output14:"+"\t"+str(pLidstone(V,S,C_x,0.10)))
    # Output 15
    output.append("Output15:"+"\t"+str(pLidstone(V,S,C_unk,0.10)))
    # Output 16
    output.append("Output16:"+"\t"+str(perplexity(Lidstone_validation, Lidstone_training_counter, V, S, 0.01)))
    # Output 17
    output.append("Output17:"+"\t"+str(perplexity(Lidstone_validation, Lidstone_training_counter, V, S, 0.10)))
    # Output 18
    output.append("Output18:"+"\t"+str(perplexity(Lidstone_validation, Lidstone_training_counter, V, S, 1.00)))
    # Output 19 & 20
    min_perplexity, min_lam = minPerplexity(Lidstone_validation, Lidstone_training_counter, V, S)
    output.append("Output19:" + "\t" + str(min_lam))
    output.append("Output20:" + "\t" + str(min_perplexity))

    ### Question 4 ###
    idx = int(round(len(words) * 0.5))
    training, held_out = words[:idx], words[idx:]
    # Output 21
    output.append("Output21:" + "\t" + str(len(training)))
    # Output 22
    output.append("Output22:" + "\t" + str(len(held_out)))
    # Output 23
    C_t, C_h, n_r, t_r = HeldOutPrep(training,held_out,V)
    output.append("Output23:" + "\t" + str(pHeldOut(C_t,n_r,t_r,len(held_out),input_word)))
    # Output 24
    output.append("Output24:" + "\t" + str(pHeldOut(C_t,n_r,t_r,len(held_out),"unseen-word")))

    ### Question 5 ###
    print LidstoneCheck(Lidstone_training_counter, V, S, 0.01)
    print HeldOutCheck(C_t, n_r, t_r, len(held_out), training, V)

    # write to output
    f = open(output_file_name, 'w')
    f.write("\n".join(output))
    f.close()

    return

def pUniform(V):
    """
    calcs uniform probability
    :param V:
    :return: uniform probability
    """
    return 1/float(V)

def parseDevelop(develop):
    """
    parses the develop file
    :param develop:
    :return: word list
    """
    with open(develop) as input_file:
        file_data = input_file.read()
    file_lines = file_data.splitlines()[1::2]
    words = ''.join(file_lines)[:-1]
    return words.split(' ')

def pLidstone(V, S, C_x, lam = 0):
    """
    calc smoothed probability lidstone
    :param V: language vocab size
    :param S: train set size
    :param C_x: counts of words
    :param lam: lambda
    :return: probability
    """
    return float(C_x + lam) / (S + (lam * V))

def LidstoneCheck(Lidstone_training_counter, V, S, lam = 0):
    """
    used to check if probabilities were calculated correctly
    :param Lidstone_training_counter: training counter
    :param V: language vocab size
    :param S: train set size
    :param lam: lambda
    :return: sum of all probabilities
    """
    n_0 = (V - len(Lidstone_training_counter)) * pLidstone(V, S, Lidstone_training_counter["unseen-word"], lam)
    n_i = [pLidstone(V, S, Lidstone_training_counter[word], lam) for word in Lidstone_training_counter]
    return sum(n_i) + n_0

def perplexity(validation, training_counter, V, S, lam):
    """
    calc perplexity for lidstone
    :param validation:
    :param training_counter:
    :param V: language vocab size
    :param S: train set size
    :param lam: lambda
    :return: perplexity
    """
    log_list = [math.log(pLidstone(V, S, training_counter[word], lam)) for word in validation]
    return math.exp(-1 * sum(log_list) / len(validation))

def frange(start, stop, step):
     """
     xrange for floats
     """
     x = start
     while x < stop:
        yield x
        x += step

def minPerplexity(Lidstone_validation, Lidstone_training_counter, V, S):
    """
    gets minimal preplexity with lambdas 0-2
    :param Lidstone_validation: validation
    :param Lidstone_training_counter: taining set
    :param V: language vocab size
    :param S: train set size
    :return: minimal perplexity and its lambda
    """
    min_perplexity = float("inf")
    min_lam = 0
    for lam in frange(0.01,2,0.01):
        cur_perplexity = perplexity(Lidstone_validation, Lidstone_training_counter, V, S, lam)
        if cur_perplexity < min_perplexity:
            min_perplexity = cur_perplexity
            min_lam = lam
    return min_perplexity, min_lam

def HeldOutPrep(training, held_out, V):
    C_t = Counter(training)
    C_h = Counter(held_out)
    n_r = Counter()
    t_r = Counter()
    # unseen words
    n_r[0] = V - len(C_t)
    unk_set = set(C_h) - set(C_t)
    for word in unk_set:
        t_r[0] += C_h[word]
    for word in C_t:
        n_r[C_t[word]] += 1
        t_r[C_t[word]] += C_h[word]
    return C_t, C_h, n_r, t_r

def pHeldOut(C_t, n_r, t_r, S_h, word):
    r = C_t[word]
    return float(t_r[r])/(S_h * n_r[r])

def HeldOutCheck(C_t, n_r, t_r, S_h, training, V):
    n_0 = (V - len(C_t)) * pHeldOut(C_t, n_r, t_r, S_h, "unseen-word")
    n_i = [pHeldOut(C_t, n_r, t_r, S_h, word) for word in C_t]
    return sum(n_i) + n_0

if __name__ == '__main__':
    main()