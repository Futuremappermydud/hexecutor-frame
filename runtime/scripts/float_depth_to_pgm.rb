#!/usr/bin/env ruby

# Convert a little-endian float32 depth/disparity tensor to a viewable PGM.
# Contrast is clipped to percentiles so isolated outliers do not flatten the
# useful image range.

if ARGV.length < 4 || ARGV.length > 8
  warn "usage: float_depth_to_pgm.rb INPUT WIDTH HEIGHT OUTPUT [X Y CROP_W CROP_H]"
  exit 2
end

input, width_s, height_s, output, *crop_args = ARGV
width = Integer(width_s)
height = Integer(height_s)
values = File.binread(input).unpack("e*")
abort "expected #{width * height} float32 values, got #{values.length}" unless values.length == width * height

finite = values.select { |value| value.finite? && value.positive? }.sort
abort "tensor contains no positive finite values" if finite.empty?
low = finite[((finite.length - 1) * 0.02).round]
high = finite[((finite.length - 1) * 0.98).round]
abort "tensor has no contrast" unless high > low

x0, y0, crop_width, crop_height = if crop_args.empty?
  [0, 0, width, height]
elsif crop_args.length == 4
  crop_args.map { |value| Integer(value) }
else
  abort "crop requires X Y CROP_W CROP_H"
end

abort "crop lies outside the tensor" if x0.negative? || y0.negative? ||
  crop_width <= 0 || crop_height <= 0 ||
  x0 + crop_width > width || y0 + crop_height > height

pixels = String.new(capacity: crop_width * crop_height, encoding: Encoding::BINARY)
(y0...(y0 + crop_height)).each do |y|
  row = y * width
  (x0...(x0 + crop_width)).each do |x|
    value = values[row + x]
    normalized = value.finite? && value.positive? ? (value - low) / (high - low) : 0.0
    pixels << [[(normalized * 255.0).round, 0].max, 255].min.chr
  end
end

File.binwrite(output, "P5\n#{crop_width} #{crop_height}\n255\n".b + pixels)
puts "wrote #{output}; 2%-98% range #{low}..#{high}"
