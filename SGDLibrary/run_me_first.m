% Add folders to path.

addpath(pwd);

cd sgd_solver/;
addpath(genpath(pwd));
cd ..;

cd problem/;
addpath(genpath(pwd));
cd ..;

cd tool/;
addpath(genpath(pwd));
cd ..;

cd plotter/;
addpath(genpath(pwd));
cd ..;

cd sgd_test/;
addpath(genpath(pwd));
cd ..;

% for GDLibrary
cd gd_solver/;
addpath(genpath(pwd));
cd ..;

cd gd_test/;
addpath(genpath(pwd));
cd ..;

cd data/;
addpath(genpath(pwd));
%cd ..;

cd ALLAML/;
addpath(genpath(pwd));
cd ..;

cd MNIST/;
addpath(genpath(pwd));
cd ..;


cd w8a/;
addpath(genpath(pwd));
cd ..;


cd Sido0/;
addpath(genpath(pwd));
cd ..;


cd cover_type.binary/;
addpath(genpath(pwd));
cd ..;


cd icnn1/;
addpath(genpath(pwd));
cd ..;

cd ..;

[version, release_date] = sgdlibrary_version();
fprintf('##########################################################\n');
fprintf('###                                                    ###\n');
fprintf('###                Welcome to SGDLibrary               ###\n');
fprintf('###    (version:%s, released:%s)     ###\n', version, release_date);
fprintf('###                                                    ###\n');
fprintf('##########################################################\n');


