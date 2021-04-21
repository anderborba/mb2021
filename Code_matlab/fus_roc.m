% Coded by Anderson Borba data: 01/07/2020 version 1.0
% Fusion of Evidences in Intensities Channels for Edge Detection in PolSAR Images 
% GRSL - IEEE Geoscience and Remote Sensing Letters 
% Anderson A. de Borba, Maurı́cio Marengoni, and Alejandro C Frery
% 
% Description (Function)
% Does the roc fusion method
% Input
%       1) E - Evidences matrix
%       2) m, n > 0 -  Matrix dimansion 
%       3) nc > 0 - channel numbers 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output 
% 1) The image fusion  
% Obs:  1) contact email: anderborba@gmail.com

function [F] = fus_roc(E, m, n, nc)
V(1: m, 1: n)  = 0.0;
M(1: m, 1: n, 1: nc) = 0.0;
for i = 1: nc
	V(:, :) = V(:, :) + E(:, :, i);
end
for i= 1: m
	for j= 1: n
		if( V(i,j) >= 1 & V(i, j) <= nc)
			M(i, j, 1) = 1;
		end
		if( V(i,j) >= 2 & V(i, j) <= nc)
	        	M(i, j, 2) = 1;
		end
		if( V(i,j) >= 3 & V(i, j) <= nc)
			M(i, j, 3) = 1;
		end
		% AAB: Modificação sugerida para o número de canais maior
		%for l = 1: nc
		%	if( V(i,j) >= l & V(i, j) <= nc)
		%		M(i, j, l) = 1;
		%	end
		%end
	end
end
dim = m * n * nc;
%   Tp1 com M1E
%   AAB: Implementação das eqs (7) no artigo g_s_fusion_2011, ou 
%   AAB: Eqs  3.62 e 3.66 no texto da tese
for l  = 1: nc
	soma(1: nc) = 0.0;
	for k  = 1: nc
		for i= 1: m
			for j= 1: n
				if( M(i, j, l) > 0 & E(i, j, k) > 0 )
					soma(k) = soma(k) + 1;
				end
			end
		end
	end
	tp(l) = sum(soma)/ dim;
end
%   Fp1 com M1E
%   AAB: Eqs  3.63 e 3.67 no texto da tese
for l  = 1: nc
	soma(1: nc) = 0.0;
	for k  = 1: nc
		for i= 1: m
			for j= 1: n
				if( M(i, j, l) > 0 & E(i, j, k) == 0 )
					soma(k) = soma(k) + 1;
				end
			end
		end
	end
	fp(l) = sum(soma)/ dim;
end
%   TN1 com M1NE
%   Fp1 com M1E
%   AAB: Eqs  3.64 e 3.68 no texto da tese
for l  = 1: nc
	soma(1: nc) = 0.0;
	for k  = 1: nc
		for i= 1: m
			for j= 1: n
				if( M(i, j, l) == 0 & E(i, j, k) == 0 )
					soma(k) = soma(k) + 1;
				end
			end
		end
	end
	tn(l) = sum(soma)/ dim;
end
%   FN1 com M1NE
%   AAB: Implementação das eqs (8) no artigo g_s_fusion_2011, ou 
%   AAB: Eqs  3.65 e 3.69 no texto da tese
for l  = 1: nc
	soma(1: nc) = 0.0;
	for k  = 1: nc
		for i= 1: m
			for j= 1: n
				if( M(i, j, l) == 0 & E(i, j, k) > 0 )
					soma(k) = soma(k) + 1;
				end
			end
		end
	end
	fn(l) = sum(soma)/ dim;
end
F = M(:,:, 2);
% AAB: Constução do gráfico ROC
% AAB: O comando imshow no programa principal tem que ser desativado,
%      para mostrar o gráfico
% AAB: O gráfico pode ser visto na figura 3.11 do texto da tese (senão quiser plot a roc)
%for i = 1: nc
%       tprj(i) =       tp(i) / (tp(i) + fn(i));
%       fprj(i) = 1 -   (tn(i) / (fp(i) + tn(i)));
        % AAB: tp + fn tem que ser igual para todos os canais
%       p(i)    = tp(i) + fn(i);
%       q(i)    = tp(i) + fp(i);
% AAB : Calcula a distancia entre o ponto e a reta diagnóstico
%       a = 1 - p(i)
%       b = p(i)
%       c = -p(i)
%       x0 = fprj(i) 
%       y0 = tprj(i) 
%       AAB: norma 2 pode ser substituido por srqt(a**2+b**2) 
%       d(i) = abs(a*x0 + b*y0 + c) / norma2(a,b)
%end
% AAB o vetor d acumula todas as distancias euclidianas
% É necessário saber qual o indíce com o valor mínimo para definir qual imagem fusão é a melhor
% EM python podemos usar o seguinte comando. 
% np.argmin(vetor) comando do numpy. Pelo que vi tem outras ideias
%paux = p(1);
%display('Valor do ponto (P,P) para a contruir a reta  diagnóstico no gráfico (ROC)');
%p;
% Gráfico 
% Habilitar para plotar o gráfico
%x =[0: paux/100: paux];
%y = ((paux - 1)/paux) * x + 1;
%plot(fprj, tprj, 'r*', x, y, 'b-');
