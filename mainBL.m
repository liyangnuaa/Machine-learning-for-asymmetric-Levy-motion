clear;
clc;

zmin=0;
zmax=5;
Nz=1e7;
z0=linspace(zmin,zmax,Nz);

% N=1;                   %%% 一个初始点给出的样本数
h=0.001;
alpha=1.5;
beta=-0.5;
sigma=0.5;
epsilong=1;

kf=6;
Kd=10;
kd=1;
Rbas=0.4;

%%% Generate data
M=h^(1/alpha)*stblrnd(alpha,beta,1,0,1,Nz);
Bh=sqrt(h)*randn(1,Nz);
xf=(6*z0.^2./(z0.^2+Kd)-kd*z0+Rbas)*h+z0./sqrt(0.5+z0.^2).*Bh+sigma*M;
xf0=z0;

%%% Identify alpha and sigma
q=epsilong;
m=5;
N=2;
nkp=zeros(1,N+1);
nkn=zeros(1,N+1);
for k=0:N
    I=((xf)>=m^k*q)&((xf)<m^(k+1)*q);
    nkp(k+1)=length(xf(I));
    I=((xf)<=-m^k*q)&((xf)>-m^(k+1)*q);
    nkn(k+1)=length(xf(I));
end
nk=nkp+nkn;
nratio=nk(1)./nk(2:end);
pos=1:N;
alpha1=log(nratio)./(pos*log(m));
alpha0=sum(alpha1)/N;
% nratio=nk(1:end-1)./nk(2:end);
% alpha1=log(nratio)./(log(m));
% alpha0=sum(alpha1)/N;

npnratio=sum(nkn)/sum(nkp);
beta0=(1-npnratio)/(1+npnratio);

kalpha=alpha0*(1-alpha0)/(cos(pi*alpha0/2)*gamma(2-alpha0));
pos=0:N;
sigmak=q*m.^pos.*(nk*alpha0/(h*Nz*kalpha*(1-m^(-alpha0)))).^(1/alpha0);
sigma0=sum(sigmak)/(N+1);

% xc=0.1:0.01:1.9;
% cnalpha=xc.*gamma((1+xc)/2)./(2.^(1-xc)*sqrt(pi).*gamma(1-xc/2));
% figure;
% plot(xc,cnalpha);
% 
% %%% Identify drift term
% Ncoef=6;
% I=(abs(xf)<epsilong);
% zinitial=xf0(I);
% x=xf(I);
% n=length(zinitial);
% A=zeros(n,Ncoef+1);
% A(:,1)=1;
% for i=1:Ncoef
%     A(:,i+1)=zinitial'.^i;
% end
% A2=A;
% B=n/Nz*x'/h;
% % X=(A'*A)\(A'*B);
% pos=0:1:Ncoef;
% for k=1:Ncoef
%     X=(A'*A)\(A'*B);
%     I=abs(X)<0.05;
%     A(:,I)=[];
%     pos(I)=[];
%     if isempty(X(I))
%         break;
%     end
% end
% 
% %%% Identify diffusion term
% A=A2;
% B=n/Nz*x.^2'/h-2*sigma0^alpha0*epsilong^(2-alpha0)*cnalpha/(2-alpha0);
% % XBt=(A'*A)\(A'*B);
% posBt=0:1:Ncoef;
% for k=1:Ncoef
%     XBt=(A'*A)\(A'*B);
%     I=abs(XBt)<0.05;
%     A(:,I)=[];
%     posBt(I)=[];
%     if isempty(XBt(I))
%         break;
%     end
% end

%%% Identify drift and diffusion terms using SSR sparsing
Ncoef=19;
I=(abs(xf)<epsilong);
zinitial=xf0(I);
x=xf(I);
n=length(zinitial);
if alpha0==1
    B=n/Nz*x'/h;
else
    B=n/Nz*x'/h-sigma0^alpha0*beta0*epsilong^(1-alpha0)*kalpha/(1-alpha0);
end
BBt=n/Nz*x.^2'/h-sigma0^alpha0*epsilong^(2-alpha0)*kalpha/(2-alpha0);
nbin=1e3;
xbin=zeros(1,nbin);
Bbin=zeros(1,nbin);
BbinBt=zeros(1,nbin);
wbin=zeros(1,nbin);
n0=floor(n/nbin);
wbin(1:end-1)=n0;
wbin(end)=(n-n0*(nbin-1));
xbin(1)=sum(zinitial(1:n0))/n0;
Bbin(1)=sum(B(1:n0))/n0;
BbinBt(1)=sum(B(1:n0))/n0;
for i=2:nbin
    I=sum(wbin(1:i-1))+1:sum(wbin(1:i));
    xbin(i)=sum(zinitial(I))/length(I);
    Bbin(i)=sum(B(I))/length(I);
    BbinBt(i)=sum(BBt(I))/length(I);
end
wbin=diag(wbin)/n0;
Dict=@(x)[x;x.^2;x.^3;sin(x);cos(11*x);sin(11*x);-10*(tanh(10*x)).^2+10;-10*(tanh(10*x-10)).^2+10;
    exp(-50*x.^2);exp(-50*(x-3).^2);exp(-0.3*(x).^2);exp(-0.3*(x-3).^2);exp(-2*(x-2).^2);exp(-2*(x-4).^2);
    exp(-50*(x-4).^2);exp(-0.6*(x-4).^2);exp(-0.6*(x-3).^2);-2*(tanh(2*x-4)).^2+2;(tanh(x-4)).^2+1];
A=Dict(xbin)';
A=[ones(nbin,1) A];
A0=wbin*A;
B0=wbin*Bbin';
B0Bt=wbin*BbinBt';

X0=(A0'*A0)\(A0'*B0);
XBt0=(A0'*A0)\(A0'*B0Bt);

%%% 设固定阈值lambda
A00=A0;
B00=B0;
B00Bt=B0Bt;
pos00=1:1:Ncoef+1;
YZ1=1.2;
YZ2=0.5;
for k=1:Ncoef
    X00=(A00'*A00)\(A00'*B00);
    I=abs(X00)<YZ1;
    A00(:,I)=[];
    pos00(I)=[];
    if isempty(X00(I))
        break;
    end
end
A00=A0;
pos00Bt=1:1:Ncoef+1;
for k=1:Ncoef
    X00Bt=(A00'*A00)\(A00'*B00Bt);
    I=abs(X00Bt)<YZ2;
    A00(:,I)=[];
    pos00Bt(I)=[];
    if isempty(X00Bt(I))
        break;
    end
end
n00=1e4;
xlin=linspace(zmin,zmax,n00);
drift=kf*xlin.^2./(xlin.^2+Kd)-kd*xlin+Rbas;
diffusion=xlin.^2./(xlin.^2+0.5);
D1=Dict(xlin)';
D1=[ones(n00,1) D1];
D2=D1(:,pos00);
drift00=D2*X00;
D2=D1(:,pos00Bt);
diffusion00=D2*X00Bt;

figure;
plot(xlin,drift);
hold on
plot(xlin,drift00);
hold off
figure;
plot(xlin,diffusion);
hold on
plot(xlin,diffusion00);
hold off


kfold=2;
nblock=floor(nbin/kfold);
n=kfold*nblock;
A=A0(1:nbin,:);
B=B0(1:nbin);
BBt=B0Bt(1:nbin);
IA=zeros(kfold,nblock);
position=1:n;
for i=1:kfold-1
    u=rand(1,nblock);
    J=kfold+1-i;
    p=0:J:J*(nblock-1);
    IA(i,:)=position(p+floor(J*u)+1);
    position(p+floor(J*u)+1)=[];
end
IA(kfold,:)=position;

A3=A;
A2=A;
B2=B;
deleteorder=zeros(1,Ncoef+1);
deltaSSR=zeros(1,Ncoef+1);
delta0=1.5;
pos=0:1:Ncoef;
for k=1:Ncoef+1
    A=A2;
    B=B2;
    X=(A'*A)\(A'*B);
    [m,I]=min(abs(X));
    deleteorder(k)=pos(I);
    A(:,I)=[];
    pos(I)=[];
    A2=A;
    
    if isempty(A)
        deltaSSR(k)=norm(B)/kfold;
        break;
    end
    
    SSR=0;
    for i=1:kfold
        A=A2;
        B=B2;
        Ai=A(IA(i,:)',:);
        Bi=B(IA(i,:)');
        A(IA(i,:)',:)=[];
        B(IA(i,:)')=[];
        SX=(A'*A)\(A'*B);
        SSR=SSR+norm(Bi-Ai*SX)^2;
    end
    deltaSSR(k)=sqrt(SSR/kfold);
    
    if (k>2)&&(deltaSSR(k)/deltaSSR(k-1)>delta0)&&(deltaSSR(k-1)/deltaSSR(k-2)<delta0)
        break;
    end
end

A2=A3;
B2=BBt;
deleteorderBt=zeros(1,Ncoef+1);
deltaSSRBt=zeros(1,Ncoef+1);
posBt=0:1:Ncoef;
for k=1:Ncoef+1
    A=A2;
    B=B2;
    XBt=(A'*A)\(A'*B);
    [m,I]=min(abs(XBt));
    deleteorderBt(k)=posBt(I);
    A(:,I)=[];
    posBt(I)=[];
    A2=A;
    
    if isempty(A)
        deltaSSRBt(k)=norm(B)/kfold;
        break;
    end
    
    SSR=0;
    for i=1:kfold
        A=A2;
        B=B2;
        Ai=A(IA(i,:)',:);
        Bi=B(IA(i,:)');
        A(IA(i,:)',:)=[];
        B(IA(i,:)')=[];
        SX=(A'*A)\(A'*B);
        SSR=SSR+norm(Bi-Ai*SX)^2;
    end
    deltaSSRBt(k)=sqrt(SSR/kfold);
    
    if (k>2)&&(deltaSSRBt(k)/deltaSSRBt(k-1)>delta0)&&(deltaSSRBt(k-1)/deltaSSRBt(k-2)<delta0)
        break;
    end
end


D1=Dict(xlin)';
D1=[ones(n00,1) D1];
D2=D1(:,pos);
driftlearn=D2*X;
D2=D1(:,posBt);
diffusionlearn=D2*XBt;

figure;
plot(xlin,drift);
hold on
plot(xlin,driftlearn);
hold off
figure;
plot(xlin,diffusion);
hold on
plot(xlin,diffusionlearn);
hold off
