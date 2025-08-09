import torch as t

class ManualSGD:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.state = {id(p): {"v": t.zeros_like(p)} for p in self.params}

    @t.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            st = self.state[id(p)]
            st["v"].mul_(self.momentum).add_(p.grad, alpha=(1 - self.momentum))
            p.add_(st["v"], alpha=-self.lr)

    @t.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

class ManualAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.state = {id(p): {"m": t.zeros_like(p), "v": t.zeros_like(p)} for p in self.params}

    @t.no_grad()
    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            st = self.state[id(p)]
            g = p.grad
            st["m"].mul_(self.b1).add_(g, alpha=1 - self.b1)
            st["v"].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)

            mhat = st["m"] / (1 - (self.b1 ** self.t))
            vhat = st["v"] / (1 - (self.b2 ** self.t))
            p.add_(mhat / (t.sqrt(vhat) + self.eps), alpha=-self.lr)

    @t.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

class ManualAdamW(ManualAdam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, lr, betas, eps)
        self.weight_decay = weight_decay

    @t.no_grad()
    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            st = self.state[id(p)]
            g = p.grad
            p.mul_(1 - self.lr * self.weight_decay)

            st["m"].mul_(self.b1).add_(g, alpha=1 - self.b1)
            st["v"].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)

            mhat = st["m"] / (1 - (self.b1 ** self.t))
            vhat = st["v"] / (1 - (self.b2 ** self.t))
            p.add_(mhat / (t.sqrt(vhat) + self.eps), alpha=-self.lr)