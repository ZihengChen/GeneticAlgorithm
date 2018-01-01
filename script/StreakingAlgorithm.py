from pylab import *

class Streaking:
    def __init__(self,
                 
                 uv_phase = np.zeros_like,
                 uv_E0    = 0.00534,#[au]
                 uv_tau   = 0.56990,#[au]
                 uv_b     = 0.29282,#[au]
                 uv_w0    = 5.51248,#[au]
                 
                 ir_phase = np.zeros_like,
                 ir_E0    = 0.00754,#[au]
                 ir_tau   = 495.868,#[au]
                 ir_w0    = 0.02683,#[au]
                 Ip       = 0.90357,#[au]
                 
                ):
        
        # global time line
        self.dtime = 0.2 #[au]
        self.rtime = 1000 #[au]
        self.time  = np.arange(-self.rtime,self.rtime,self.dtime) #[au]
        
        # xuv constant
        self.uv_phase = uv_phase
        self.uv_E0    = uv_E0
        self.uv_tau   = uv_tau
        self.uv_a     = 2*np.log(2)/(uv_tau**2)
        self.uv_b     = uv_b
        self.uv_w0    = uv_w0
        
        # ir constant
        self.ir_phase = ir_phase
        self.ir_E0    = ir_E0
        self.ir_tau   = ir_tau
        self.ir_w0    = ir_w0
        
        # atomic constant
        self.Ip = Ip
        
        # Model time-dependent variables
        self.uv_recoE = self.UV_RecoE()
        self.ir_recoE = self.IR_RecoE()
        self.ir_recoA = self.IR_RecoA()
        self.ir_recoPHI1 = self.IR_RecoPHI1()
        self.ir_recoPHI2 = self.IR_RecoPHI2()

    
    def Integrate(self,arr):
        temp = []
        integration = 0
        for n in range(arr.size):
            integration += arr[n]
            temp.append(integration)
        temp = np.array(temp)
        temp = temp*self.dtime
        return temp
        
    def UV_RecoE (self):
        t     = self.time
        Gamma = self.uv_a - 1j * self.uv_b
        phase = self.uv_phase(t)
        temp  = self.uv_E0 * np.exp(-Gamma*t**2) * np.exp(1j*self.uv_w0*t+phase)
        temp  = temp.real
        return temp
    
    def IR_RecoE(self):
        t = self.time
        phase = self.ir_phase(t)
        temp = self.ir_E0 * np.exp(-2*np.log(2)*(t/self.ir_tau)**2) * np.cos(self.ir_w0*t+phase)
        temp = temp.real
        return temp

    def IR_RecoA(self):
        temp = - self.Integrate(self.ir_recoE)
        return temp
    
    def IR_RecoPHI1(self):
        # integral in [t,+inf]
        whole = np.sum(self.ir_recoA)*self.dtime
        temp  = whole - self.Integrate(self.ir_recoA)
        #temp = self.Integrate(self.ir_recoA)
        return temp
    
    def IR_RecoPHI2(self):
        # integral in [t,+inf]
        whole = np.sum(self.ir_recoA)*self.dtime
        temp  = whole - self.Integrate(0.5*self.ir_recoA**2)
        #temp = self.Integrate(0.5*self.ir_recoA**2)
        return temp
    
    # momentum delay dependent part
    def IR_RecoPHI(self,p):
        return p*self.ir_recoPHI1 + self.ir_recoPHI2
    
    def DipoleMoment(self,p):
        a = 2*self.Ip
        pi= np.pi
        return p/a/(pi*a)**0.75 * np.exp(- p**2/(2*a))
    
    def Signal_Phase(self, p):
        t = self.time
        return (0.5*p**2 + self.Ip)*t - self.IR_RecoPHI(p)
    
    def Signal_uv_E_shift(self,tau):
        t = self.time
        shift = int(tau//self.dtime)
        uv_E_shift = np.zeros_like(self.uv_recoE)
        if shift   > 0:
            uv_E_shift[:-shift] = self.uv_recoE[shift:]
        elif shift < 0:
            uv_E_shift[-shift:] = self.uv_recoE[:shift]
        else:
            uv_E_shift = self.uv_recoE
        return uv_E_shift
        
    def Signal_Reco(self,p,tau):
        t = self.time
        Eshift   = self.Signal_uv_E_shift(tau)
        dipole   = self.DipoleMoment(p+self.ir_recoA)
        phase    = self.Signal_Phase(p)
        
        realpart = Eshift*dipole*np.cos(phase)
        imagpart = Eshift*dipole*np.sin(phase)
        
        realpart = np.sum(realpart)*self.dtime
        imagpart = np.sum(imagpart)*self.dtime
        
        return realpart**2+imagpart**2
    
    def CalStreaking(self,parr,tauarr):
        t = self.time
        Eshift   = np.array([self.Signal_uv_E_shift(tau)
                             for tau in tauarr])
        dipole   = np.array([self.DipoleMoment(p+self.ir_recoA)
                             for p in parr])
        phase    = np.array([self.Signal_Phase(p)
                             for p in parr])
        
        realpart = np.dot(dipole*np.cos(phase),Eshift.T) * self.dtime
        imagpart = np.dot(dipole*np.sin(phase),Eshift.T) * self.dtime
        streaking= realpart**2+imagpart**2
        return streaking