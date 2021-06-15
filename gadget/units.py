import astropy.units as u


GAMMA = 5. / 3
M_UNIT = 1e10 * u.Msun / u.littleh
R_UNIT = u.Mpc / u.littleh
V_UNIT = u.km / u.s * 1 / u.littleh
T_UNIT = u.K
RHO_UNIT = M_UNIT / R_UNIT ** 3
DIMENSIONLESS = u.dimensionless_unscaled
UNKOWN = u.dimensionless_unscaled

UNITS = {
    'Coordinates': R_UNIT,
    'Mass': M_UNIT,
    'Velocity': V_UNIT,
    'ParticleIDs': DIMENSIONLESS,
    'Particle_Binding_Energy': V_UNIT**2,
    'Density': RHO_UNIT,
    'InternalEnergy': V_UNIT**2,
    'OnEquationOfState': DIMENSIONLESS,
    'Hydrogen': DIMENSIONLESS,
    'Helium': DIMENSIONLESS,
    'Carbon': DIMENSIONLESS,
    'Nitrogen': DIMENSIONLESS,
    'Oxygen': DIMENSIONLESS,
    'Neon': DIMENSIONLESS,
    'Magnesium': DIMENSIONLESS,
    'Silicon': DIMENSIONLESS,
    'Iron': DIMENSIONLESS,
    'SmoothedMetallicity': DIMENSIONLESS,
    'SmoothingLength': R_UNIT,
    'StarFormationRate': M_UNIT / u.yr,
    'Temperature': T_UNIT,
    'InitialMass': M_UNIT,
    'StellarFormationTime': DIMENSIONLESS,
    'BH_BirthTime': DIMENSIONLESS,
    'BH_CumlAccrMass': M_UNIT,
    'BH_CumlSeedMass': M_UNIT,
    'BH_EnergyReservoir': V_UNIT**2,
    'BH_Mass': M_UNIT,
    'BH_Mdot': M_UNIT / u.yr,
    'GroupNumber': DIMENSIONLESS,
    'HostHalo_TVir': T_UNIT,
    'HostHalo_TVir_Mass': T_UNIT,
    'IronFromSNIa': DIMENSIONLESS,
    'Metallicity': DIMENSIONLESS,
    'SmoothedIronFromSNIa': DIMENSIONLESS,
    'SubGroupNumber': DIMENSIONLESS,
    'ContaminationCount': DIMENSIONLESS,
    'ContaminationMass': M_UNIT,
    'FirstSubhaloID': DIMENSIONLESS,
    'GroupCentreOfPotential': R_UNIT,
    'GroupLength': DIMENSIONLESS,
    'GroupMass': M_UNIT,
    'GroupOffset': DIMENSIONLESS,
    'Group_M_Crit200': M_UNIT,
    'Group_M_Crit2500': M_UNIT,
    'Group_M_Crit500': M_UNIT,
    'Group_M_Mean200': M_UNIT,
    'Group_M_Mean2500': M_UNIT,
    'Group_M_Mean500': M_UNIT,
    'Group_M_TopHat200': M_UNIT,
    'Group_R_Crit200': R_UNIT,
    'Group_R_Crit2500': R_UNIT,
    'Group_R_Crit500': R_UNIT,
    'Group_R_Mean200': R_UNIT,
    'Group_R_Mean2500': R_UNIT,
    'Group_R_Mean500': R_UNIT,
    'Group_R_TopHat200': R_UNIT,
    'NumOfSubhalos': DIMENSIONLESS,
    'CentreOfMass': M_UNIT,
    'CentreOfPotential': R_UNIT,
    'GasSpin': R_UNIT * V_UNIT,
    'GroupNumber': DIMENSIONLESS,
    'HalfMassProjRad': R_UNIT,
    'HalfMassRad': R_UNIT,
    'IDMostBound': DIMENSIONLESS,
    'InertiaTensor': M_UNIT * R_UNIT**2,
    'InitialMassWeightedBirthZ': DIMENSIONLESS,
    'InitialMassWeightedStellarAge': R_UNIT / V_UNIT,
    'KineticEnergy': V_UNIT**2,
    'Mass': M_UNIT,
    'MassType': M_UNIT,
    'Mass_001kpc': M_UNIT,
    'Mass_003kpc': M_UNIT,
    'Mass_005kpc': M_UNIT,
    'Mass_010kpc': M_UNIT,
    'Mass_020kpc': M_UNIT,
    'Mass_030kpc': M_UNIT,
    'Mass_040kpc': M_UNIT,
    'Mass_050kpc': M_UNIT,
    'Mass_070kpc': M_UNIT,
    'Mass_100kpc': M_UNIT,
    'MassWeightedEntropy': V_UNIT**2 * (M_UNIT / R_UNIT**3)**(-GAMMA + 1),
    'MassWeightedTemperature': T_UNIT,
    'MetalMass': M_UNIT,
    'MetalMassSmoothed': M_UNIT,
    'Spin': R_UNIT * V_UNIT,
    'Parent': DIMENSIONLESS,
    'StellarInitialMass': M_UNIT,
    'StellarVelDisp': V_UNIT,
    'StellarVelDisp_HalfMassProjRad': V_UNIT,
    'SubLength': DIMENSIONLESS,
    'SubOffset': DIMENSIONLESS,
    'ThermalEnergy': V_UNIT**2,
    'TotalEnergy': V_UNIT**2,
    'Velocity': V_UNIT,
    'Vmax': V_UNIT,
    'VmaxRadius': V_UNIT,
}
