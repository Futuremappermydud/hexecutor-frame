#!/usr/bin/env ruby

# Reproject a 616x616 little-endian float depth map into an OpenVR-oriented PLY.
# +X is right, +Y is up, and -Z is forward. An optional local-support threshold
# can remove isolated speckles when converting an unfiltered depth capture.

abort "usage: depth_to_ply.rb DEPTH_F32 LUMA_U8 OUTPUT.ply [MAX_DEPTH_M [MIN_NEIGHBORS]]" unless (3..5).cover?(ARGV.length)

depth_path, luma_path, output_path, max_depth_arg, support_arg = ARGV
width = 616
height = 616
focal = 380.0
center_x = 308.0
center_y = 308.0
minimum_depth = 0.25
maximum_depth = max_depth_arg ? Float(max_depth_arg) : 5.0
minimum_support = support_arg ? Integer(support_arg) : 0

depth = File.binread(depth_path).unpack("e*")
luma = File.binread(luma_path).bytes
expected = width * height
abort "expected #{expected} depth values, got #{depth.length}" unless depth.length == expected
abort "expected #{expected} luma bytes, got #{luma.length}" unless luma.length == expected

vertices = []
(1...(height - 1)).each do |y|
  (1...(width - 1)).each do |x|
    index = y * width + x
    z = depth[index]
    next unless z.finite? && z >= minimum_depth && z <= maximum_depth

    tolerance = [0.04, z * 0.04].max
    support = 0
    (-1..1).each do |dy|
      (-1..1).each do |dx|
        next if dx.zero? && dy.zero?
        neighbor = depth[(y + dy) * width + x + dx]
        support += 1 if neighbor.finite? && neighbor.positive? &&
          (neighbor - z).abs <= tolerance
      end
    end
    next if support < minimum_support

    point_x = (x - center_x) * z / focal
    point_y = -(y - center_y) * z / focal
    point_z = -z
    intensity = luma[index]
    vertices << [point_x, point_y, point_z, intensity, intensity, intensity]
  end
end

header = <<~PLY
  ply
  format binary_little_endian 1.0
  comment Frame Spatial Runtime confidence-filtered Vulkan stereo
  comment coordinates: +X right, +Y up, -Z forward; units meters
  element vertex #{vertices.length}
  property float x
  property float y
  property float z
  property uchar red
  property uchar green
  property uchar blue
  end_header
PLY

File.open(output_path, "wb") do |file|
  file.write(header)
  vertices.each { |vertex| file.write(vertex.pack("e3C3")) }
end

puts "wrote #{output_path} with #{vertices.length} supported metric points"
