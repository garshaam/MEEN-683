%% HIGH LEVEL WORKFLOW

% Guess Initial Target RPM 
% Calc Q,T,P from BEMT
% Compute I  = Q/Kt
% Determine Motor Term Voltage 
% Compute new RPM sag
% Repeat until convergence on SS RPM 



%% PARAM

Kv = 2300; % RPM/Volt
Voc = 22.2; % 6s Lipo Voltage
Rm = 0.5; %ohm
Rbatt = 0.05; %ohm for 6 cell lipo



function Vterm = DroneBatt(Voc, I, Rbatt)
Vterm = Voc - I*Rbatt;
end

function [RPM,I] = DroneMotor(Q, Vterm, MotorKv, Rm)

Kt = 60/(2*pi*MotorKv);

I = Q / Kt;

RPM = MotorKv*(Vterm - I*Rm);

end

