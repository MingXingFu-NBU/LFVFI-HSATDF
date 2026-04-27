function output = RangeCompressor(input)

mu = 5000;
output = log(1 + mu * input) / log(1 + mu);

end