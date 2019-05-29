# カードをランダムに配る．01の配列で出力

import numpy as np
import random


class Dealing:

    def __init__(self):
        self.deal_array = np.array(range(52))
        self.deal = {}   # 手札が0~52の数字ランダム順に13枚ずつ
        self.player = ['N', 'E', 'S', 'W']
        self.pbn_style = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.pbn = ''
        self.hand = {}  # 手札が52の0, 1で表現されている. Club 2~A, Diamond 2~A, Heart 2~A, Spade 2~A

    def deal_card(self):
        random.shuffle(self.deal_array)
        self.deal['N'] = self.deal_array[:13]
        self.deal['E'] = self.deal_array[13:26]
        self.deal['S'] = self.deal_array[26:39]
        self.deal['W'] = self.deal_array[39:]

    def deal_opponent_again(self):
        deal_opponent = np.hstack((self.deal['E'], self.deal['W']))
        random.shuffle(deal_opponent)   # 敵の配り直し
        self.deal['E'] = deal_opponent[:13]
        self.deal['W'] = deal_opponent[13:26]

    def convert_style(self):
        for i in self.player:
            self.hand[i] = np.array([0]*52)

        self.pbn = 'N:'

        for p in self.player:
            for i in self.deal[p]:
                self.hand[p][i] = 1

        for p in self.player:
            cards = [''] * 4
            for suit in range(4):
                for i in range(12, -1, -1):
                    if self.hand[p][i + (suit * 13)] == 1:
                        cards[suit] += self.pbn_style[i]
            self.pbn += cards[3] + '.' + cards[2] + '.' + cards[1] + '.' + cards[0] + ' '

    def print_pbn(self):
        print(self.pbn)

    def print_hand(self):
        print(self.hand)

    def return_pbn(self):
        return self.pbn

    def return_hand(self):
        return self.hand



if __name__ == '__main__':
    deal = Dealing()
    deal.deal_card()
    deal.convert_style()
    deal.print_pbn()
    deal.print_hand()

    deal.deal_opponent_again()
    deal.convert_style()
    deal.print_pbn()
    deal.print_hand()

    deal.deal_opponent_again()
    deal.convert_style()
    deal.print_pbn()
    deal.print_hand()

    deal.deal_opponent_again()
    deal.convert_style()
    deal.print_pbn()
    deal.print_hand()


#    with open('memo.txt', mode = 'w') as f:
#        for i in range(1000000):
#            pbn, hand = deal()
            # f.write('#'+str(i))
#            f.write(pbn+'\n')
            # f.write(str(hand['N'])+'\n')
            # f.write(str(hand['E'])+'\n')
            # f.write(str(hand['S'])+'\n')
            # f.write(str(hand['W'])+'\n')
#            print(i)
