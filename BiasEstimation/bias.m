load('orientation_1.mat')

accx1 = Acceleration.X(:,1);
accy1 = Acceleration.Y(:,1);
accz1 = Acceleration.Z(:,1);

load('orientation_2.mat')

accx2 = Acceleration.X(:,1);
accy2 = Acceleration.Y(:,1);
accz2 = Acceleration.Z(:,1);


bx = (accx1 + accx2)/2;
by = (accy1 + accy2)/2;
bz = ((accz1 + accz2)/2) - 0.1;

bx_avg = sum(bx)/604;
by_avg = sum(by)/604;
bz_avg = sum(bz)/604;

T = table(['bias_x';'bias_y';'bias_z'],[bx_avg; by_avg; bz_avg]);
writetable(T, 'bias.csv')
