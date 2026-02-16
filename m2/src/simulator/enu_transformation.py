import math

A = 6378137.0
ONE_F = 298.257223563
B = A * (1.0 - 1.0 / ONE_F)
E2 = (1.0 / ONE_F) * (2.0 - (1.0 / ONE_F))


def NN(p):
    return A / math.sqrt(1.0 - (E2) * math.sin(p) * math.sin(p))


class ENU:
    def __init__(
        self,
        origin_longitude=139.759771,
        origin_latitude=35.712964,
        origin_altitude=23.4,
        origin_geoid_height=38.2263,
    ):
        self.origin_longitude = origin_longitude
        self.origin_latitude = origin_latitude
        self.origin_altitude = origin_altitude
        self.origin_geoid_height = origin_geoid_height
        self.input_origin_point()
        self.initialize_mat()

    def input_origin_point(self):
        self.base_lo = self.origin_longitude * math.pi / 180.0
        self.base_la = self.origin_latitude * math.pi / 180.0
        self.base_al = self.origin_altitude
        self.base_geoid_height = self.origin_geoid_height

    def lla2ecef(self, lo, la, al):
        h = al
        ret_x = (NN(la) + h) * math.cos(la) * math.cos(lo)
        ret_y = (NN(la) + h) * math.cos(la) * math.sin(lo)
        ret_z = (NN(la) * (1.0 - E2) + h) * math.sin(la)
        return ret_x, ret_y, ret_z

    def initialize_mat(self):
        self.base_x, self.base_y, self.base_z = self.lla2ecef(
            self.base_lo, self.base_la, self.base_al + self.base_geoid_height
        )

        b11 = math.cos(math.pi / 2.0)
        b12 = math.sin(math.pi / 2.0)
        b13 = 0.0
        b21 = -math.sin(math.pi / 2.0)
        b22 = math.cos(math.pi / 2.0)
        b23 = 0.0
        b31 = 0.0
        b32 = 0.0
        b33 = 1.0

        c11 = math.cos(math.pi / 2.0 - self.base_la)
        c12 = 0.0
        c13 = -math.sin(math.pi / 2.0 - self.base_la)
        c21 = 0.0
        c22 = 1.0
        c23 = 0.0
        c31 = math.sin(math.pi / 2.0 - self.base_la)
        c32 = 0.0
        c33 = math.cos(math.pi / 2.0 - self.base_la)

        d11 = math.cos(self.base_lo)
        d12 = math.sin(self.base_lo)
        d13 = 0.0
        d21 = -math.sin(self.base_lo)
        d22 = math.cos(self.base_lo)
        d23 = 0.0
        d31 = 0.0
        d32 = 0.0
        d33 = 1.0

        # e = b*c
        e11 = b11 * c11 + b12 * c21 + b13 * c31
        e12 = b11 * c12 + b12 * c22 + b13 * c32
        e13 = b11 * c13 + b12 * c23 + b13 * c33

        e21 = b21 * c11 + b22 * c21 + b23 * c31
        e22 = b21 * c12 + b22 * c22 + b23 * c32
        e23 = b21 * c13 + b22 * c23 + b23 * c33

        e31 = b31 * c11 + b32 * c21 + b33 * c31
        e32 = b31 * c12 + b32 * c22 + b33 * c32
        e33 = b31 * c13 + b32 * c23 + b33 * c33

        # a = b*c*d = e*d
        self.a11 = e11 * d11 + e12 * d21 + e13 * d31
        self.a12 = e11 * d12 + e12 * d22 + e13 * d32
        self.a13 = e11 * d13 + e12 * d23 + e13 * d33

        self.a21 = e21 * d11 + e22 * d21 + e23 * d31
        self.a22 = e21 * d12 + e22 * d22 + e23 * d32
        self.a23 = e21 * d13 + e22 * d23 + e23 * d33

        self.a31 = e31 * d11 + e32 * d21 + e33 * d31
        self.a32 = e31 * d12 + e32 * d22 + e33 * d32
        self.a33 = e31 * d13 + e32 * d23 + e33 * d33

    def calc_enu_based_on_altitude_including_geoid_height(self, lo, la, al):
        self.longitude = lo * math.pi / 180.0
        self.latitude = la * math.pi / 180.0
        self.altitude = al

        buf_x, buf_y, buf_z = self.lla2ecef(self.longitude, self.latitude, self.altitude)
        delta_x = buf_x - self.base_x
        delta_y = buf_y - self.base_y
        delta_z = buf_z - self.base_z

        ret_e = self.a11 * delta_x + self.a12 * delta_y + self.a13 * delta_z  # e
        ret_n = self.a21 * delta_x + self.a22 * delta_y + self.a23 * delta_z  # n
        ret_u = self.a31 * delta_x + self.a32 * delta_y + self.a33 * delta_z  # u

        return ret_e, ret_n, ret_u


if __name__ == "__main__":
    enu = ENU(origin_longitude=139.759771, origin_latitude=35.712964, origin_altitude=23.4)
    e, n, u = enu.calc_enu_based_on_altitude_including_geoid_height(139.759771, 35.712964, 23.4)
    print(f"Including geoid height: e={e}, n={n}, u={u}")
