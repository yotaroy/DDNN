from dds import functions
from dds import dds
import ctypes
import numpy as np
from dealing_cards import Dealing
import time


def DDS():
    tableDealPBN = dds.ddTableDealPBN()
    table = dds.ddTableResults()
    myTable = ctypes.pointer(table)
    dds.SetMaxThreads(0)

    deal = Dealing()

    deal.deal_card()
    deal.convert_style()
    # print(deal.return_pbn())

    hand = deal.return_hand()   # 手札が52の0, 1で表現されている. Club 2~A, Diamond 2~A, Heart 2~A, Spade 2~A. [[N],[E],[S],[W]]

    # DDSのためにhandの情報をpbn形式で入れる
    tableDealPBN.cards = deal.return_pbn().encode('utf-8')

    # DDSの実行
    dds.CalcDDtablePBN(tableDealPBN, myTable)

    # DDSの結果の収納
    dds_result = np.array(functions.dds_result(myTable))  # C:0 D:1 H:2 S:3 NT:4, N:0 E:1 S:2 W:3,  result[suit][declarer]
    # print(result)

    # print(tableDealPBN.cards)
    # print(result)
    # print('N result average')
    result = dict()
    result['N'] = np.round([dds_result[i][0] for i in range(5)])
    result['E'] = np.round([dds_result[i][1] for i in range(5)])
    result['S'] = np.round([dds_result[i][2] for i in range(5)])
    result['W'] = np.round([dds_result[i][3] for i in range(5)])


    # N, E, S, W
    return np.stack((hand['N'], hand['E'], hand['S'], hand['W'])), np.stack((result['N'], result['E'], result['S'], result['W']))

if __name__ == "__main__":
    t = time.time()
    for _ in range(10):
        print(DDS())
    print(time.time() - t)
