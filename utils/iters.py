import math

class plotCounter:
    def __init__(self, layout, totalsize, overlap ):
        self.current = -1
        self.currentonround = -1
        self.totalsize  = totalsize
        self.overlap    = overlap
        self.layout     = layout
        self.groupsize  = self.layout[0] * self.layout[1]
        self.rounds     = math.ceil((self.totalsize)/(self.groupsize-self.overlap))
        self.currentround = 0
        self.totalspots = self.groupsize*self.rounds
        self.totalplots = (self.rounds-1)*self.overlap + self.totalsize
        self.emptyspots = self.totalspots - self.totalplots
        if self.rounds == 1:
            self.groupsize=self.totalspots
        if self.emptyspots+self.overlap == self.groupsize:
            self.rounds -= 1
            self.totalspots = self.groupsize*self.rounds
            self.totalplots = (self.rounds-1)*self.overlap + self.totalsize
            self.emptyspots = self.totalspots - self.totalplots

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        self.currentonround += 1
        if self.currentonround==self.groupsize:
            self.currentround+=1
            self.current -= self.overlap
            self.currentonround=0
        if self.current < self.totalsize and self.currentround < self.rounds:
            return self.current, self.currentonround, self.currentround
        raise StopIteration

class plotRound:
    def __init__(self, layout, totalsize, overlap, round):
        self.totalsize  = totalsize
        self.overlap    = overlap
        self.layout     = layout
        self.groupsize  = self.layout[0] * self.layout[1]
        if self.groupsize==1: self.overlap=0
        self.current = (self.groupsize*round -1)-(self.overlap*round)
        self.currentonround = -1
        self.rounds     = math.ceil((self.totalsize)/(self.groupsize-self.overlap))
        self.currentround = round
        if self.rounds == 1:
            self.groupsize=self.totalspots
    
    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        self.currentonround += 1
        if self.currentonround==self.groupsize:
            raise StopIteration
        if self.current < self.totalsize and self.currentround < self.rounds:
            return self.current, self.currentonround, self.currentround
        else:
            return None, self.currentonround, self.currentround

