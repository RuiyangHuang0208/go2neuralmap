import pygame
import sys

pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    # 获取当前所有按键状态
    keys = pygame.key.get_pressed()
    
    # 检测特定按键
    if keys[pygame.K_LEFT]:
        print("向左移动")
    if keys[pygame.K_RIGHT]:
        print("向右移动")
    if keys[pygame.K_UP]:
        print("向上移动")
    if keys[pygame.K_DOWN]:
        print("向下移动")
    
    # 字母和数字键
    if keys[pygame.K_SPACE]:
        print("空格键按下")
    if keys[pygame.K_a]:
        print("A键按下")
    if keys[pygame.K_q]:
        print("q键按下")
        exit()
    if keys[pygame.K_1]:
        print("数字1按下")
    
    screen.fill((0, 0, 0))
    clock.tick(60)
