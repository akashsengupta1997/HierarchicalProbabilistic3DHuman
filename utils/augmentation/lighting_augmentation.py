import torch


def augment_light_t(batch_size, device, loc_r_range=(0.05, 3.0)):
    """
    Samples batch of random point light locations.
    Azimuth/elevation is uniformly sampled over unit sphere.
        This is done by uniform sampling a location on the unit
        sphere surface via normalised Gaussian random vector.
    Distance (r) is uniformly sampled in r_range.
    :param loc_r_range: light distance range
    :return: light_t: (B, 3)
    """
    direction = torch.randn(batch_size, 3, device=device)
    direction = direction / torch.norm(direction, dim=-1)

    l, h = loc_r_range
    r = (h - l) * torch.rand(batch_size, device=device) + l

    light_t = direction * r
    return light_t


def augment_light_colour(batch_size, device,
                         ambient_intensity_range=(0.2, 0.8),
                         diffuse_intensity_range=(0.2, 0.8),
                         specular_intensity_range=(0.2, 0.8)):
    """
    Samples batch of random light INTENSITIES (not colours because
    I am forcing RGB components to be equal i.e. white lights).
    :param ambient_intensity_range: ambient component intensity range
    :param diffuse_intensity_range: diffuse component intensity range
    :param specular_intensity_range: specular component intensity range
    :return:
    """

    l, h = ambient_intensity_range
    ambient = (h - l) * torch.rand(batch_size, device=device) + l
    ambient = ambient[:, None].expand(-1, 3)

    l, h = diffuse_intensity_range
    diffuse = (h - l) * torch.rand(batch_size, device=device) + l
    diffuse = diffuse[:, None].expand(-1, 3)

    l, h = specular_intensity_range
    specular = (h - l) * torch.rand(batch_size, device=device) + l
    specular = specular[:, None].expand(-1, 3)

    return ambient, diffuse, specular


def augment_light(batch_size,
                  device,
                  rgb_augment_config):
    light_t = augment_light_t(batch_size=batch_size,
                              device=device,
                              loc_r_range=rgb_augment_config.LIGHT_LOC_RANGE)
    ambient, diffuse, specular = augment_light_colour(batch_size=batch_size,
                                                      device=device,
                                                      ambient_intensity_range=rgb_augment_config.LIGHT_AMBIENT_RANGE,
                                                      diffuse_intensity_range=rgb_augment_config.LIGHT_DIFFUSE_RANGE,
                                                      specular_intensity_range=rgb_augment_config.LIGHT_SPECULAR_RANGE)
    lights_settings = {'location': light_t,
                       'ambient_color': ambient,
                       'diffuse_color': diffuse,
                       'specular_color': specular}
    return lights_settings






