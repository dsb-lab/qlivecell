import math


class CyclicList:
    def __init__(self, items):
        self.items = items
        if len(self.items) == 0:
            raise IndexError("CyclicList is empty")

    def __getitem__(self, index):
        normalized_index = index % len(self.items)
        return self.items[normalized_index]

    def __len__(self):
        return len(self.items)

    def get_control(self, item):
        return (item % len(self.items)) / (len(self) - 1)

    def get_map(self):
        item_list = []
        output_list = []
        for item in range(len(self)):
            item_list.append(self.get_control(item))
            output_list.append(self[item])
        return item_list, output_list


class plotCounter:
    def __init__(self, layout, totalsize, overlap):
        self.current = -1
        self.currentonround = -1
        self.totalsize = totalsize
        self.overlap = overlap
        self.layout = layout
        self.groupsize = self.layout[0] * self.layout[1]
        self.rounds = math.ceil((self.totalsize) / (self.groupsize - self.overlap))
        self.currentround = 0
        self.totalspots = self.groupsize * self.rounds
        self.totalplots = (self.rounds - 1) * self.overlap + self.totalsize
        self.emptyspots = self.totalspots - self.totalplots
        if self.rounds == 1:
            self.groupsize = self.totalspots
        if self.emptyspots + self.overlap == self.groupsize:
            self.rounds -= 1
            self.totalspots = self.groupsize * self.rounds
            self.totalplots = (self.rounds - 1) * self.overlap + self.totalsize
            self.emptyspots = self.totalspots - self.totalplots

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        self.currentonround += 1
        if self.currentonround == self.groupsize:
            self.currentround += 1
            self.current -= self.overlap
            self.currentonround = 0
        if self.current < self.totalsize and self.currentround < self.rounds:
            return self.current, self.currentonround, self.currentround
        raise StopIteration


class plotRound:
    def __init__(self, layout, totalsize, overlap, round):
        self.totalsize = totalsize
        self.overlap = overlap
        self.layout = layout
        self.groupsize = self.layout[0] * self.layout[1]
        if self.groupsize == 1:
            self.overlap = 0
        self.current = (self.groupsize * round - 1) - (self.overlap * round)
        self.currentonround = -1
        self.rounds = math.ceil((self.totalsize) / (self.groupsize - self.overlap))
        self.currentround = round
        if self.rounds == 1:
            self.groupsize = totalsize
        first, last = self.get_first_and_last_in_round(round)
        if last - first < (self.groupsize - 1):
            self.current = last - self.groupsize + 1

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        self.currentonround += 1
        if self.currentonround == self.groupsize:
            raise StopIteration
        if self.current < self.totalsize and self.currentround < self.rounds:
            return self.current, self.currentonround, self.currentround
        else:
            return None, self.currentonround, self.currentround

    def get_first_and_last_in_round(self, cr):
        first = (self.groupsize * cr) - (self.overlap * cr) + 1
        last = first + self.groupsize - 1
        last = min(last, self.totalsize)
        return first, last
