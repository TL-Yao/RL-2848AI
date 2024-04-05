import pygame


class Renderer:
    def __init__(self, size):
        self.screen_size = 600
        self.grid_size = self.screen_size // size
        self.screen = None
        self.font = None
        self.game_over_font = None
        self.colors = {
            0: (204, 192, 179),
            1: (238, 228, 218),
            2: (237, 224, 200),
            3: (242, 177, 121),
            4: (245, 149, 99),
            5: (246, 124, 95),
            6: (246, 94, 59),
            7: (237, 207, 114),
            8: (237, 204, 97),
            9: (237, 200, 80),
            10: (237, 197, 63),
            11: (237, 194, 46),
            12: (93, 218, 216),
            13: (60, 131, 238),
            14: (0, 0, 238),
            15: (128, 0, 128),
            16: (81, 0, 81),
            17: (0, 0, 0),
        }

    def render(self, board, score, illegal_move, move, end):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.screen_size, self.screen_size + 140)
            )
            pygame.display.set_caption("2048")
            self.font = pygame.font.Font(None, 40)
            self.game_over_font = pygame.font.Font(None, 72)

        self.screen.fill((187, 173, 160))
        grid_color = (204, 192, 179)

        for i in range(4):
            for j in range(4):
                pygame.draw.rect(
                    self.screen,
                    grid_color,
                    (
                        j * self.grid_size,
                        i * self.grid_size,
                        self.grid_size,
                        self.grid_size,
                    ),
                )
                pygame.draw.rect(
                    self.screen,
                    (187, 173, 160),
                    (
                        j * self.grid_size + 5,
                        i * self.grid_size + 5,
                        self.grid_size - 10,
                        self.grid_size - 10,
                    ),
                )

        for i in range(4):
            for j in range(4):
                value = board[i][j]
                if value > 0:
                    color = self.colors[int(value * 16)]
                    pygame.draw.rect(
                        self.screen,
                        color,
                        (
                            j * self.grid_size + 5,
                            i * self.grid_size + 5,
                            self.grid_size - 10,
                            self.grid_size - 10,
                        ),
                    )
                    value_str = str(2 ** (int(value * 16)))
                    text = self.font.render(value_str, True, (0, 0, 0))
                    text_rect = text.get_rect(
                        center=(
                            j * self.grid_size + self.grid_size // 2,
                            i * self.grid_size + self.grid_size // 2,
                        )
                    )
                    self.screen.blit(text, text_rect)

        text_y = self.screen_size + 20

        text = self.font.render(f"Score: {score}", True, (0, 0, 0))
        text_rect = text.get_rect(topleft=(20, text_y))
        self.screen.blit(text, text_rect)

        text = self.font.render(f"Move: {move}", True, (0, 0, 0))
        text_rect = text.get_rect(topleft=(20, text_y + 40))
        self.screen.blit(text, text_rect)

        text = self.font.render(f"Illegal Move: {illegal_move}", True, (0, 0, 0))
        text_rect = text.get_rect(topleft=(20, text_y + 80))
        self.screen.blit(text, text_rect)

        if end:
            overlay = pygame.Surface(
                (self.screen_size, self.screen_size), pygame.SRCALPHA
            )
            overlay.fill((255, 255, 255, 128))  # 白色，50% 透明度
            self.screen.blit(overlay, (0, 0))
            text = self.game_over_font.render("Game Over", True, (255, 0, 0))
            text_rect = text.get_rect(
                center=(self.screen_size // 2, self.screen_size // 2)
            )
            self.screen.blit(text, text_rect)

        pygame.display.flip()
