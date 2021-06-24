\begin{equation}
f_{Z}(z; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}}\exp\{-(z-\mu)^2/2\sigma^2\}
(\#eq:01)
  \end{equation}

#$$
#\begin{equation}
#\ell(z; \mu, \sigma)=\log\frac{1}{\sigma} + \log\frac{1}{\sqrt{2\pi}} + \log\Bigg(\exp\{-(z-\mu)^2/2\sigma^2\}\Bigg)
#\end{equation}
#$$
#$$
#\begin{equation}
#\ell(z; \mu, \sigma)=\log\sigma^{-1} + \log\sqrt{2\pi}^{-1}-\frac{1}{2\sigma^2}(z-\mu)^2
#\end{equation}
#$$
#$$
#\begin{equation}
#\ell(z; \mu, \sigma)=-\log\sigma - \log\sqrt{2\pi}-\frac{1}{2\sigma^2}(z-\mu)^2
#\end{equation}
#$$
#$$
#\begin{equation}
#\mathcal{L}(\mu, \sigma)= \sum_{i=1}^n\ell(z_i; \mu, \sigma)
#\end{equation}
#$$
#$$
#\begin{equation}
#\mathcal{L}(\mu, \sigma)= \sum_{i=1}^n\Bigg[-\log\big(\sigma\big) - #\log\big(\sqrt{2\pi}\big)-\frac{1}{2\sigma^2}(z_i-\mu)^2\Bigg]
#\end{equation}
#$$
#$$
#\begin{equation}
#\mathcal{L}(\mu, \sigma)= -\sum_{i=1}^n\log\big(\sigma\big) - #\sum_{i=1}^n\log\big(\sqrt{2\pi}\big)-\frac{1}{2\sigma^2}\sum_{i=1}^n\Bigg[\frac{1}{2\sigma^2}(z_i-\mu)^2\Bigg]
#\end{equation}
#$$

#$$
#\begin{equation}
#\mathcal{L}(\mu, \sigma)= -n\log\big(\sigma\big) - n\log\big(\sqrt{2\pi}\big)-\frac{1}{2\sigma^2}\sum_{i=1}^n(z_i-\mu)^2
#\end{equation}
#$$