function dx=ddeModel(t,x,xdelay,theta)
dx = theta.fc3.Weights*tanh(theta.fc2.Weights...
    *tanh(theta.fc1.Weights*[x;xdelay]+theta.fc1.Bias)...
    +theta.fc2.Bias);
end