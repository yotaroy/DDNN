import numpy as np

class Score:
    def __init__(self, vul=False):
        self.minor = 20
        self.major = 30
        self.NT = 10
        self.make = 50
        self.val = vul
        if vul:
            self.game = 450
            self.small_slam = 750
            self.grand_slam = 750
            self.down = -100
        else:
            self.game = 250
            self.small_slam = 500
            self.grand_slam = 500
            self.down = -50

        zero_array = np.array([0]*13)
        one_array = np.array([1]*13)
        self.HCP_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4] * 4)
        self.Spade_array = np.hstack((one_array, zero_array, zero_array, zero_array))
        self.Heart_array = np.hstack((zero_array, one_array, zero_array, zero_array))
        self.Diamond_array = np.hstack((zero_array, zero_array, one_array, zero_array))
        self.Club_array = np.hstack((zero_array, zero_array, zero_array, one_array))

    def calc_score(self, suit, contract, tricks):    # suit: 0-4, contract: 1-7, trick: 0-13    contract:0 = pass
        if contract == 0:   # PASSでコントラクトなし
            point = 0
        elif tricks < contract + 6:
            point = self.down * (contract + 6 - tricks)
        else:
            if suit <= 1:   # C, D
                point = self.minor * (tricks - 6)
            elif suit <= 3:     # H, S
                point = self.major * (tricks - 6)
            else:   # NT
                point = self.major * (tricks - 6) + self.NT

            point += self.make

            if contract >= 5 or (contract >= 4 and suit >= 2) or (contract >= 3 and suit == 4):
                point += self.game
                if contract >= 6:
                    point += self.small_slam
                    if contract == 7:
                        point += self.grand_slam

        return point

    def calc_dds_bid_score(self, bid, dds):  # bid: 0~35, dds: numpy 5dim: S0, H1, D2, C3, NT4, ddsに関しては変更多数なので注意！！！！！！
        if bid == 0:
            return 0

        contract = (bid+4) // 5
        suit = (bid-1) % 5  # C0, D1, H2, S3, NT4
        if suit == 4:
            tricks = int(dds[4])
        else:
            tricks = int(dds[3-suit])

        return self.calc_score(suit, contract, tricks)

    def calc_dds_bid_score_max(self, dds): # bid: 0~35, dds: numpy 5dim: S0, H1, D2, C3, NT4    (optimal action)
        suit = [3, 2, 1, 0, 4]
        max_score = 0
        for i in range(5):
            if dds[i] >= 7:
                max_score = max(max_score, self.calc_score(suit[i], dds[i]-6, dds[i]))
        return max_score

    def calc_reward(self, dds):  # ダブルド、リダブルドは考慮しない ddsはtensor

        dds_int = dds.astype(int)
        score = np.zeros(36)

        for suit in range(5): # C D H S NT
            for contract in range(1, 8):
                point = self.calc_score(suit, contract, dds_int[suit].item())

                score[suit-4+contract*5] = point

        return score.astype(int)

    def score_to_imp(self, player_score, opponent_score):
        if player_score > opponent_score:
            win = 1
        else:
            win = -1

        difference_point = abs(player_score - opponent_score)

        if difference_point < 20:       # 0- 20
            IMPs = 0
        elif difference_point < 50:     # 20-40
            IMPs = 1
        elif difference_point < 90:     # 50-80
            IMPs = 2
        elif difference_point < 130:    # 90-120
            IMPs = 3
        elif difference_point < 170:    # 130-160
            IMPs = 4
        elif difference_point < 220:    # 170-210
            IMPs = 5
        elif difference_point < 270:    # 220-260
            IMPs = 6
        elif difference_point < 320:    # 270-310
            IMPs = 7
        elif difference_point < 370:    # 320-360
            IMPs = 8
        elif difference_point < 430:    # 370-420
            IMPs = 9
        elif difference_point < 500:    # 430-490
            IMPs = 10
        elif difference_point < 600:    # 500-590
            IMPs = 11
        elif difference_point < 750:    # 600-740
            IMPs = 12
        elif difference_point < 900:    # 750-890
            IMPs = 13
        elif difference_point < 1100:   # 900-1090
            IMPs = 14
        elif difference_point < 1300:   # 1100-1290
            IMPs = 15
        elif difference_point < 1500:   # 1300-1490
            IMPs = 16
        elif difference_point < 1750:   # 1500-1740
            IMPs = 17
        elif difference_point < 2000:   # 1750-1900
            IMPs = 18
        elif difference_point < 2250:   # 2000-2240
            IMPs = 19
        elif difference_point < 2500:   # 2250-2490
            IMPs = 20
        elif difference_point < 3000:   # 2500-2990
            IMPs = 21
        elif difference_point < 3500:   # 3000-3490
            IMPs = 22
        elif difference_point < 4000:   # 3500-3990
            IMPs = 23
        else:                           # 4000+
            IMPs = 24

        return IMPs*win

    def feature_extraction(self, hand): # hand: numpy形式　52次元の0, 1で表現されている. Spade 2~A, Heart 2~A, Diamond 2~A, Club 2~A
        HCP = np.sum(hand * self.HCP_array)      # High Card Points
        S = np.sum(hand * self.Spade_array)      # num of spades
        H = np.sum(hand * self.Heart_array)      # num of hearts
        D = np.sum(hand * self.Diamond_array)    # num of diamonds
        C = np.sum(hand * self.Club_array)       # num of clubs

        return HCP, S, H, D, C


