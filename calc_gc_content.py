def calculate_gc_content(pwm):
  if isinstance(pwm, dict):
    length_of_motif = len(pwm['A'])
    total_gc_content = 0

    for i in range(length_of_motif):
        gc_content_at_position = pwm['G'][i] + pwm['C'][i]
        total_gc_content += gc_content_at_position

    average_gc_content = total_gc_content / length_of_motif
    
  elif isinstance(pwm, list):
    length_of_motif = len(pwm[0])
    total_gc_content = 0

    for i in range(length_of_motif):
        gc_content_at_position = pwm[2][i] + pwm[1][i]
        total_gc_content += gc_content_at_position

    average_gc_content = total_gc_content / length_of_motif

  else:
    raise TypeError("PWM must be a dictionary or a list")
    
  return average_gc_content

# # Example usage
# pwm = {
#     'A': [0.1, 0.2, 0.3, 0.1],
#     'C': [0.2, 0.3, 0.1, 0.2],
#     'G': [0.3, 0.1, 0.2, 0.3],
#     'T': [0.4, 0.4, 0.4, 0.4]
# }

# print(calculate_gc_content(pwm) == 0.425)

# pwm = [
#     [0.1, 0.2, 0.3, 0.1],
#     [0.2, 0.3, 0.1, 0.2],
#     [0.3, 0.1, 0.2, 0.3],
#     [0.4, 0.4, 0.4, 0.4]
# ]

# print(calculate_gc_content(pwm) == 0.425)

# pwm = (
#     [0.1, 0.2, 0.3, 0.1],
#     [0.2, 0.3, 0.1, 0.2],
#     [0.3, 0.1, 0.2, 0.3],
#     [0.4, 0.4, 0.4, 0.4]
# )

# print(calculate_gc_content(pwm))
