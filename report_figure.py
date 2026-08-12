"""report_figure.py -- assembles Figure 1 of the report from results.json.
Run experiments.py first; writes figs/figMain.pdf."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":7.5,"axes.grid":True,"grid.alpha":.25,
 "savefig.bbox":"tight","axes.spines.top":False,"axes.spines.right":False,
 "legend.frameon":False,"lines.linewidth":1.1,"pdf.fonttype":42})
C=["#1f4e79","#c0392b","#2e8b57","#8e44ad","#d68910"]
R=json.load(open("results.json"))
fig,ax=plt.subplots(2,3,figsize=(9.4,4.4))
E1=R["E1"]; dts=np.array(E1["dts"])
for i,sg in enumerate(E1["sigmas"]):
    ax[0,0].loglog(dts,E1["rmse"][str(sg)],color=C[i],label=rf"$\sigma={sg:g}$")
    ax[0,0].axvline(E1["h_opt_theory"][i],color=C[i],ls=":",lw=.9)
ax[0,0].set_xlabel(r"step size $h=\Delta t$");ax[0,0].set_ylabel(r"RMSE of $\hat{\dot x}$")
ax[0,0].set_title("(a) central differences",fontsize=8);ax[0,0].legend(fontsize=6)
sg=E1["sigmas"]
ax[0,1].loglog(sg,E1["h_opt_empirical"],"o",color=C[0],label="empirical $h^*$")
ax[0,1].loglog(sg,E1["h_opt_theory"],"-",color=C[0],label=r"$(3\sigma/M_3)^{1/3}$")
ax[0,1].loglog(sg,E1["min_err"],"s",color=C[1],label="empirical min RMSE")
ax[0,1].loglog(sg,np.array(E1["min_err"])[0]*(np.array(sg)/sg[0])**(2/3),"--",color=C[1],label=r"$\propto\sigma^{2/3}$")
ax[0,1].set_xlabel(r"noise level $\sigma$");ax[0,1].set_ylabel("value")
ax[0,1].set_title("(b) optimum and error floor",fontsize=8);ax[0,1].legend(fontsize=6)
lab={"exact_deriv":r"exact $\dot X$ (noisy library)","fd":"finite differences","savgol":"Savitzky--Golay"}
for i,m in enumerate(["exact_deriv","fd","savgol"]):
    d=R["E2"]["lorenz"][m]; s=np.array(d["sigmas"],float); s[0]=3e-5
    ax[0,2].semilogx(s,d["rate"],"o-",color=C[i],ms=2.5,label=lab[m])
ax[0,2].set_xlabel(r"$\sigma_{\rm rel}$");ax[0,2].set_ylabel("support recovery rate");ax[0,2].set_ylim(-.05,1.05)
ax[0,2].set_title("(c) Lorenz, by estimator",fontsize=8);ax[0,2].legend(fontsize=6)
lams=np.array(R["E3"]["lams"])
for i,s0 in enumerate(R["E3"]["show"]):
    c=R["E3"]["curves"][str(s0)]
    ax[1,0].semilogx(lams,c["nact"],color=C[i],label=rf"$\sigma_{{\rm rel}}={s0:g}$")
    if c["lam_lo"]: ax[1,0].axvspan(c["lam_lo"],c["lam_hi"],color=C[i],alpha=.08)
    ax[1,1].loglog(lams,np.maximum(c["err"],1e-16),color=C[i])
ax[1,0].axhline(R["E3"]["n_true"],color="k",ls="--",lw=.9,label="true no. of terms")
ax[1,0].set_xlabel(r"threshold $\lambda$");ax[1,0].set_ylabel("active terms")
ax[1,0].set_title(r"(d) recovery window in $\lambda$",fontsize=8);ax[1,0].legend(fontsize=6)
ax[1,1].set_xlabel(r"threshold $\lambda$");ax[1,1].set_ylabel("relative coefficient error")
ax[1,1].set_title(r"(e) accuracy vs. $\lambda$",fontsize=8)
nm={"oscillator":"oscillator","lorenz":"Lorenz","lotka":"Lotka--Volterra"}
for i,k in enumerate(["oscillator","lorenz","lotka"]):
    d=R["E2"][k]["savgol"]; s=np.array(d["sigmas"],float); s[0]=3e-5
    ax[1,2].semilogx(s,d["rate"],"o-",color=C[i],ms=2.5,label=nm[k])
ax[1,2].axhline(.5,color="k",ls=":",lw=.8)
ax[1,2].set_xlabel(r"$\sigma_{\rm rel}$");ax[1,2].set_ylabel("recovery rate");ax[1,2].set_ylim(-.05,1.05)
ax[1,2].set_title(r"(f) $\sigma_c$ by system",fontsize=8);ax[1,2].legend(fontsize=6)
fig.tight_layout(pad=0.4);fig.savefig("figs/figMain.pdf");print("ok")
